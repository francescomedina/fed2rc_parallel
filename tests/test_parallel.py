"""Run: NUMBA_NUM_THREADS=4 python -m pytest -q

Set FED2RC_CPP_SOLVER to a compiled benchmark_solver executable to also check
the independent C++ implementation; compilation is intentionally not implicit.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from numba import get_num_threads, set_num_threads, threading_layer
from sktime.transformations.panel.rocket import Rocket
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))
from parallel_utils import FeatureCache, select_lambda, transform_features
from benchmark_solver import lu_solve, make_problem, search


def original_transform(X, seeds, ppv_only):
    features = []
    for seed in seeds:
        rocket = Rocket(num_kernels=1, random_state=seed, n_jobs=1)
        rocket.fit(np.zeros((1, 1, X.shape[1])))
        out = rocket.transform(X[:, None, :]).to_numpy()
        features.append(out[:, :1] if ppv_only else out)
    return np.concatenate(features, axis=1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize("ppv_only", [True, False])
def test_rocket_parity_and_thread_count(dtype, ppv_only):
    X = (np.random.default_rng(8).normal(size=(13, 37)) * 3).astype(dtype)
    X[0] = 2  # Constant series also exercises normalization's epsilon.
    seeds = [7, 0, 13, 7, 99]
    expected = original_transform(X, seeds, ppv_only)
    previous = get_num_threads()
    try:
        for threads in (1, min(previous, 4)):
            set_num_threads(threads)
            actual = transform_features(X, seeds, ppv_only=ppv_only, caching=False)
            np.testing.assert_array_equal(actual, expected)
    finally:
        set_num_threads(previous)


def test_cache_identity_order_and_eviction():
    X = np.random.default_rng(13).normal(size=(8, 31))
    cache = FeatureCache(max_bytes=8 * 4 * 2, max_entries=2)
    expected = transform_features(X, [1, 2, 1], caching=False)
    actual = transform_features(X, [1, 2, 1], cache=cache)
    np.testing.assert_array_equal(actual, expected)
    actual[:] = -123  # Returned arrays cannot corrupt cached features.
    np.testing.assert_array_equal(transform_features(X.copy(), [1, 2, 1], cache=cache), expected)
    X[0, 0] += 100  # Same object, changed content must invalidate its identity.
    np.testing.assert_array_equal(transform_features(X, [1, 2, 3], cache=cache),
                                  transform_features(X, [1, 2, 3], caching=False))
    np.testing.assert_array_equal(transform_features(X, [1], ppv_only=False, cache=cache),
                                  original_transform(X, [1], False))
    assert cache.nbytes <= cache.max_bytes
    cache.clear()
    assert cache.nbytes == 0
    assert transform_features(X, [], cache=cache).shape == (8, 0)
    np.testing.assert_array_equal(transform_features(X[0], [1], caching=False),
                                  original_transform(X[:1], [1], True))


def test_cache_concurrent_publication():
    cache = FeatureCache(max_bytes=4096, max_entries=16)
    values = [np.full((17, 1), i, dtype=np.float32) for i in range(8)]

    def worker(i):
        for _ in range(30):
            cache.put(i, values[i])
            result = cache.get_many([i])[0]
            np.testing.assert_array_equal(result, values[i])
            with pytest.raises(ValueError):
                result.setflags(write=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(8)))
    assert cache.nbytes == sum(value.nbytes for value in values)


def test_warm_cache_does_not_transform(monkeypatch):
    X = np.random.default_rng(9).normal(size=(8, 32))
    cache = FeatureCache()
    expected = transform_features(X, [1, 2, 3], cache=cache)

    def unexpected_compute(*args):
        pytest.fail("A cache hit must not launch the numerical transform")

    monkeypatch.setattr("parallel_utils._transform_map", unexpected_compute)
    np.testing.assert_array_equal(transform_features(X.copy(), [3, 1, 3], cache=cache),
                                  expected[:, [2, 0, 2]])


def test_concurrent_transforms():
    X = np.random.default_rng(4).normal(size=(8, 32))
    expected = transform_features(X, [1, 3, 5], caching=False)  # Compile first.
    if threading_layer() == "workqueue":
        pytest.skip("Numba workqueue does not support concurrent parallel callers")
    cache = FeatureCache()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(transform_features, X, [1, 3, 5], cache=cache)
                   for _ in range(4)]
        for future in futures:
            np.testing.assert_array_equal(future.result(), expected)


def original_search(A, b, X, y, lambdas):
    scores = []
    for lam in lambdas:
        W = np.linalg.solve(A + lam * np.eye(len(A)), b)
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-10
        scores.append(np.mean(np.argmax(X @ W, axis=1) == y))
    return np.array(scores)


@pytest.mark.parametrize("indefinite", [False, True])
def test_lambda_parity(indefinite):
    rng = np.random.default_rng(19)
    Q = rng.normal(size=(7, 7))
    A = Q.T @ Q
    if indefinite:
        A -= 10 * np.eye(7)  # Compression/subtraction need not preserve SPD.
    b, X = rng.normal(size=(7, 3)), rng.normal(size=(23, 7)).astype(np.float32)
    y = rng.integers(0, 3, len(X))
    lambdas = np.logspace(-3, 0, 23)[::-1]  # Tie rule must not rely on sorting.
    expected = original_search(A, b, X, y, lambdas)
    previous = get_num_threads()
    try:
        for threads in (1, min(previous, 4)):
            set_num_threads(threads)
            lam, acc, scores = select_lambda(A, b, X, y, lambdas)
            np.testing.assert_array_equal(scores, expected)
            best = max(range(len(lambdas)), key=lambda i: (expected[i], lambdas[i]))
            assert (lam, acc) == (lambdas[best], expected[best])
    finally:
        set_num_threads(previous)
    assert all(pool["num_threads"] == 1 for pool in threadpool_info()
               if pool["user_api"] == "blas")


def test_zero_accuracy_ties_and_invalid_system():
    A, b, X = np.eye(3), np.zeros((3, 2)), np.ones((5, 3))
    lam, acc, scores = select_lambda(A, b, X, np.ones(5, dtype=int), [1, 0.01, 0.1])
    assert (lam, acc) == (1.0, 0.0)
    np.testing.assert_array_equal(scores, np.zeros(3))
    with pytest.raises(ValueError):
        select_lambda(A, b, X, np.ones(5), [])
    with pytest.raises(np.linalg.LinAlgError, match="index 1"):
        select_lambda(-np.eye(3), b, X, np.ones(5), [0.01, 1.0, 0.1])


def test_benchmark_lu_against_lapack():
    A, b, X, y, lambdas = make_problem(9, 7)
    for lam in lambdas:
        np.testing.assert_allclose(lu_solve(A, b, lam),
                                   np.linalg.solve(A + lam * np.eye(len(A)), b),
                                   rtol=1e-12, atol=1e-14)
    # The SPD fixture rarely pivots. Explicitly test a nonsingular matrix that
    # requires row interchanges, including permutation of multiple RHS columns.
    A = np.array([[0., 2., 1.], [1., -2., 0.], [3., 1., 4.]])
    b = np.arange(9, dtype=float).reshape(3, 3)
    np.testing.assert_allclose(lu_solve(A, b, 0.0), np.linalg.solve(A, b), atol=1e-13)


def test_cpp_python_benchmark():
    executable = os.environ.get("FED2RC_CPP_SOLVER")
    if not executable:
        pytest.skip("Set FED2RC_CPP_SOLVER to check compiled C++ benchmark")
    arguments = ["24", "13", "2", "2"]
    cpp = json.loads(subprocess.check_output([executable, *arguments], text=True))
    python = json.loads(subprocess.check_output(
        [sys.executable, str(ROOT / "benchmark_solver.py"), *arguments], text=True))
    for key in ("best_index", "best_lambda", "accuracy", "scores"):
        assert cpp[key] == python[key]
    np.testing.assert_allclose(cpp["solution"], python["solution"], rtol=1e-12, atol=1e-14)
    assert cpp["residual"] < 1e-12 and python["residual"] < 1e-12
