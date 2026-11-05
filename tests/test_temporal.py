"""Temporal partitions versus untouched kernel-only maps and sktime kernels.

Run with NUMBA_NUM_THREADS=4 NUMBA_THREADING_LAYER=omp OMP_NUM_THREADS=4.
Both backends must preserve their own original floating-point arithmetic.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numba
import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifier
from sktime.transformations.rocket import Rocket
from sktime.transformations.rocket._rocket_numba import _apply_kernel_univariate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import openmp_utils as cpp
import parallel_utils as nb
from temporal import resolve_temporal_chunks


def prepared(samples=3, timepoints=33, kernels=5, *, integers=False):
    rng = np.random.default_rng(2048 + timepoints)
    generate = (lambda shape: rng.integers(-3, 4, shape).astype(np.float32)) if integers else (
        lambda shape: rng.normal(size=shape).astype(np.float32)
    )
    X, weights = generate((samples, timepoints)), generate((kernels, 11))
    lengths = np.resize(np.array([1, 3, 7, 9, 11], np.int32), kernels)
    dilations = np.resize(np.array([0, 1, 2, 3, 5], np.int32), kernels)
    # Alternating unpadded/padded kernels, including kernels wider than input.
    paddings = np.maximum(0, ((lengths - 1) * dilations + 1 - timepoints + 1) // 2)
    paddings[1::2] = ((lengths[1::2] - 1) * dilations[1::2]) // 2
    biases = generate(kernels)
    return X, weights, lengths, biases, dilations, paddings.astype(np.int32)


def raw(backend, data, chunks, threads=4, ppv_only=False, return_partials=False):
    X, weights, lengths, biases, dilations, paddings = data
    if backend == "cpp":
        return cpp.native.transform(X, weights, biases, lengths, dilations, paddings,
                                    ppv_only, threads, chunks)
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
        out = np.empty((len(lengths), len(X), 1 if ppv_only else 2), np.float32)
        if chunks == 1:
            nb._transform_map(X, weights, lengths, biases, dilations, paddings, out)
        else:
            counts = np.empty((len(lengths), len(X), chunks), np.int64)
            maxima = np.empty(counts.shape, np.float32)
            nb._transform_temporal(X, weights, lengths, biases, dilations, paddings,
                                   out, counts, maxima)
            if return_partials:
                return out, counts, maxima
        return np.concatenate(list(out), axis=1)
    finally:
        numba.set_num_threads(previous)


@pytest.mark.parametrize("backend", ["cpp", "numba"])
@pytest.mark.parametrize("timepoints", [2, 3, 7, 8, 15, 16, 31, 32, 101, 102, 8193])
@pytest.mark.parametrize("chunks", [2, 4, 7])
def test_exact_partition_equivalence(backend, timepoints, chunks):
    data = prepared(timepoints=timepoints)
    expected = raw(backend, data, 1, threads=1)
    for threads in (1, 2, 4):
        actual = raw(backend, data, chunks, threads=threads)
        np.testing.assert_array_equal(actual, expected)
    # Exact reference using identical arithmetic and kernels for the Numba path.
    if backend == "numba":
        X, weights, lengths, biases, dilations, paddings = data
        for k in range(len(lengths)):
            for row in range(len(X)):
                ppv, maximum = _apply_kernel_univariate(
                    X[row], weights[k, :lengths[k]], lengths[k], biases[k],
                    dilations[k], paddings[k],
                )
                np.testing.assert_array_equal(expected[row, k * 2:k * 2 + 2], [ppv, maximum])


@pytest.mark.parametrize("backend", ["cpp", "numba"])
@pytest.mark.parametrize("padding", [0, 1, 5, 20])
@pytest.mark.parametrize("dilation", [0, 1, 2, 7])
def test_halo_and_partition_seams(backend, padding, dilation):
    # Exactly represented products let us compare both implementations to an
    # independent scalar oracle without compiler-specific contraction effects.
    X = np.zeros((2, 17), np.float32)
    X[0, [0, 7, 8, 9, 16]] = [2, 3, 7, -2, 5]
    X[1] = -2
    length = 3
    positions = X.shape[1] + 2 * padding - (length - 1) * dilation
    if positions < 1:
        return
    weights = np.zeros((1, 11), np.float32)
    weights[0, :3] = [1, -2, 1]
    data = (X, weights, np.array([length], np.int32), np.array([1], np.float32),
            np.array([dilation], np.int32), np.array([padding], np.int32))
    expected = []
    for x in X:
        responses = []
        for q in range(positions):
            value = 1.0
            for tap in range(length):
                pos = q - padding + tap * dilation
                if 0 <= pos < len(x):
                    value += float(weights[0, tap]) * float(x[pos])
            responses.append(value)
        expected.append([np.count_nonzero(np.array(responses) > 0) / positions, max(responses)])
    for chunks in (2, 4, positions + 1):
        np.testing.assert_array_equal(raw(backend, data, chunks), np.array(expected, np.float32))


@pytest.mark.parametrize("backend", ["cpp", "numba"])
def test_unequal_partitions_use_counts_not_mean_of_ppvs(backend):
    weights = np.zeros((1, 11), np.float32)
    weights[0, 0] = 1
    data = (np.array([[1, 1, -1, -1, -1]], np.float32), weights,
            np.array([1], np.int32), np.array([0], np.float32),
            np.array([1], np.int32), np.array([0], np.int32))
    np.testing.assert_array_equal(raw(backend, data, 2), np.array([[0.4, 1]], np.float32))
    if backend == "numba":
        _, counts, _ = raw(backend, data, 2, return_partials=True)
        np.testing.assert_array_equal(counts, [[[2, 0]]])


@pytest.mark.parametrize("backend", ["cpp", "numba"])
@pytest.mark.parametrize("ppv_only", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_end_to_end_feature_extraction(backend, ppv_only, dtype):
    rng = np.random.default_rng(21)
    X = (rng.normal(size=(9, 82)) * 5).astype(dtype)[:, ::2]
    X[0] = 3
    before = X.copy()
    seeds = [11, 2, 11, 0, 7]
    transform = cpp.transform_features if backend == "cpp" else nb.transform_features
    expected = transform(X, seeds, ppv_only=ppv_only, caching=False, temporal_chunks=1)
    for chunks in (2, 4, 9, "auto"):
        actual = transform(X, seeds, ppv_only=ppv_only, caching=False, temporal_chunks=chunks)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float32
    original = []
    for seed in seeds:
        rocket = Rocket(num_kernels=1, random_state=seed, n_jobs=1).fit(np.zeros((1, 1, 41)))
        values = rocket.transform(X[:, None, :]).to_numpy()
        original.append(values[:, :1] if ppv_only else values)
    reference = np.concatenate(original, axis=1)
    np.testing.assert_array_equal(expected[:, ::1 if ppv_only else 2], reference[:, ::1 if ppv_only else 2])
    if not ppv_only:
        np.testing.assert_allclose(expected[:, 1::2], reference[:, 1::2], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(X, before)


def test_cache_independent_of_temporal_execution_strategy():
    X = np.random.default_rng(7).normal(size=(4, 513))
    cache = nb.FeatureCache()
    expected = nb.transform_features(X, [3, 7, 3], temporal_chunks=1, cache=cache)
    for chunks in (2, 4, "auto"):
        np.testing.assert_array_equal(
            nb.transform_features(X.copy(), [7, 3, 7], temporal_chunks=chunks, cache=cache),
            expected[:, [1, 0, 1]],
        )
        np.testing.assert_array_equal(
            nb.transform_features(X, [3, 7, 3], temporal_chunks=chunks, caching=False), expected,
        )


@pytest.mark.parametrize("backend", ["cpp", "numba"])
def test_predictions_unchanged(backend):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(24, 259))
    y = np.arange(24) % 3
    transform = cpp.transform_features if backend == "cpp" else nb.transform_features
    baseline = transform(X, range(12), temporal_chunks=1, caching=False)
    classifier = RidgeClassifier().fit(baseline[:18], y[:18])
    for chunks in (2, 4, "auto"):
        combined = transform(X, range(12), temporal_chunks=chunks, caching=False)
        np.testing.assert_array_equal(combined, baseline)
        np.testing.assert_array_equal(classifier.predict(combined[18:]), classifier.predict(baseline[18:]))


@pytest.mark.parametrize("backend", ["cpp", "numba"])
def test_concurrent_callers_have_private_scratch(backend):
    data = prepared(samples=2, timepoints=8195, kernels=3)
    expected = raw(backend, data, 1, threads=1)
    raw(backend, data, 4, threads=2)  # Compile before concurrent calls.
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: raw(backend, data, 4, threads=2), range(4)))
    for result in results:
        np.testing.assert_array_equal(result, expected)
    assert not np.shares_memory(results[0], results[1])


@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, "2", "bad", None])
@pytest.mark.parametrize("backend", ["cpp", "numba"])
def test_invalid_temporal_chunks(invalid, backend):
    transform = cpp.transform_features if backend == "cpp" else nb.transform_features
    with pytest.raises(ValueError, match="temporal_chunks"):
        transform(np.ones((2, 13)), [0], temporal_chunks=invalid)


@pytest.mark.parametrize("backend", ["cpp", "numba"])
def test_empty_seed_list(backend):
    transform = cpp.transform_features if backend == "cpp" else nb.transform_features
    for chunks in (1, 2, 4, "auto"):
        assert transform(np.ones((2, 13)), [], temporal_chunks=chunks).shape == (2, 0)


def test_auto_avoids_short_or_saturated_work():
    lengths = np.array([7], np.int32)
    dilation = np.array([1], np.int32)
    padding = np.array([3], np.int32)
    assert resolve_temporal_chunks("auto", 128, lengths, dilation, padding, 4, 1) == 1
    assert resolve_temporal_chunks("auto", 100_000, lengths, dilation, padding, 1, 1) == 1
    assert resolve_temporal_chunks("auto", 100_000, lengths, dilation, padding, 4, 4) == 1
    assert resolve_temporal_chunks("auto", 100_000, lengths, dilation, padding, 4, 1) == 4
    assert resolve_temporal_chunks("auto", 100_000, np.repeat(lengths, 2),
                                   np.repeat(dilation, 2), np.repeat(padding, 2), 4, 2) == 2
    assert resolve_temporal_chunks(8, 2, np.array([1]), dilation, np.array([0]), 4, 1) == 2
    assert resolve_temporal_chunks("auto", 16384, np.array([1]), dilation, np.array([0]), 4, 1) == 1
    assert resolve_temporal_chunks("auto", 1024, np.repeat(lengths, 8),
                                   np.repeat(dilation, 8), np.repeat(padding, 8),
                                   4, 2, samples=64) == 2


@pytest.mark.parametrize("backend", ["cpp", "numba"])
def test_one_series_one_kernel_really_uses_multiple_workers(backend):
    threads = min(4, numba.config.NUMBA_NUM_THREADS)
    if threads < 2:
        pytest.skip("run with NUMBA_NUM_THREADS>=2 to verify temporal concurrency")
    data = prepared(samples=1, timepoints=65537, kernels=1)
    X, weights, lengths, biases, dilations, paddings = data
    worker_ids = np.full(threads, -1, np.int32)
    if backend == "cpp":
        cpp.native.transform(X, weights, biases, lengths, dilations, paddings,
                             False, threads, threads, worker_ids)
    else:
        out = np.empty((1, 1, 2), np.float32)
        counts = np.empty((1, 1, threads), np.int64)
        maxima = np.empty(counts.shape, np.float32)
        previous = numba.get_num_threads()
        try:
            numba.set_num_threads(threads)
            nb._transform_temporal(X, weights, lengths, biases, dilations, paddings,
                                   out, counts, maxima, worker_ids)
        finally:
            numba.set_num_threads(previous)
    np.testing.assert_array_equal(np.unique(worker_ids), np.arange(threads))


@pytest.mark.parametrize("backend", ["cpp", "numba"])
def test_random_geometry_fuzz(backend):
    rng = np.random.default_rng(128)
    for _ in range(30):
        t = int(rng.integers(2, 300))
        X, w, lengths, biases, dilation, padding = prepared(2, t, 7)
        lengths[:] = rng.integers(1, 12, 7)
        dilation[:] = rng.integers(0, 16, 7)
        padding[:] = np.maximum(rng.integers(0, 80, 7), ((lengths - 1) * dilation - t + 2) // 2)
        data = X, w, lengths, biases, dilation, padding
        expected = raw(backend, data, 1, threads=1)
        np.testing.assert_array_equal(raw(backend, data, int(rng.integers(2, 10))), expected)


@pytest.mark.parametrize("backend", ["cpp", "numba"])
def test_cancellation_zero_threshold_and_maximum_bits(backend):
    # Reordering the dot product could turn a zero response into a positive one.
    # Temporal partitioning must not change that dot product, even with fastmath.
    X = np.ones((2, 67), np.float32)
    X[1, 1::2] = np.nextafter(np.float32(1), np.float32(2))
    weights = np.zeros((4, 11), np.float32)
    weights[0, :3] = [1e8, 1, -1e8]
    weights[1, :3] = [1.0000001, -1, -1e-7]
    weights[2, :3] = [1e20, -1e20, 1e-20]
    weights[3, :3] = [-0.0, -0.0, -0.0]
    data = (X, weights, np.full(4, 3, np.int32), np.array([0, 0, 0, -0.0], np.float32),
            np.array([1, 2, 3, 1], np.int32), np.array([1, 0, 3, 0], np.int32))
    expected = raw(backend, data, 1, threads=1)
    for chunks in (2, 4, 9):
        actual = raw(backend, data, chunks)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
