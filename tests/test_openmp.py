"""Regression against the untouched implementation, not a second parallel copy.

All numerical inputs and ROCKET seeds are fixed. Thread counts change scheduling
only; the real-data checks compare predictions on the official held-out split.
"""

import ast
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from sktime.transformations.rocket import Rocket


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from openmp_utils import select_lambda, transform_features  # noqa: E402


@pytest.fixture(scope="session")
def original_transform():
    """Execute the original two definitions without importing application code."""
    source = subprocess.run(
        ["git", "show", "209009c55b6b4d84b748ac5043ab48ee568fe4e5:code/utils.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    definitions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in {"RocketKernel", "transform_seeds"}
    ]
    namespace = {"np": np, "Rocket": Rocket}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "baseline_utils.py", "exec"), namespace)
    return namespace


def reference_lambda(A, b, X, y, lambdas):
    """Original LU -> column normalization -> first argmax -> accuracy loop."""
    scores = []
    for lam in lambdas:
        weights = np.linalg.solve(A + lam * np.eye(A.shape[0]), b)
        weights = weights / (np.linalg.norm(weights, axis=0, keepdims=True) + 1e-10)
        predictions = np.argmax(X @ weights, axis=1)
        scores.append(np.mean(predictions == y))
    scores = np.asarray(scores)
    best = max(range(len(lambdas)), key=lambda i: (scores[i], lambdas[i]))
    return float(lambdas[best]), float(scores[best]), scores


def check_lambda_result(actual, expected):
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    np.testing.assert_array_equal(actual[2], expected[2])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("ppv_only", [False, True])
def test_rocket_matches_original(original_transform, dtype, strided, ppv_only):
    rng = np.random.default_rng(381)
    source = rng.normal(size=(11, 78)).astype(dtype)
    X = source[:, ::2] if strided else source[:, :39].copy()
    before = X.copy()
    seeds = [17, 2, 17, 0, 91, 4]  # Reordering and duplicate columns are observable.
    if ppv_only:
        expected = original_transform["transform_seeds"](X, seeds, X.shape[1])
    else:
        kernel_type = original_transform["RocketKernel"]
        expected = np.concatenate(
            [kernel_type(seed, X.shape[1], ppv_only=False).transform(X) for seed in seeds],
            axis=1,
        )

    serial = transform_features(X, seeds, ppv_only=ppv_only, threads=1)
    parallel = transform_features(X, seeds, ppv_only=ppv_only, threads=4)
    assert parallel.dtype == np.float32
    assert parallel.shape == expected.shape
    np.testing.assert_array_equal(X, before)
    np.testing.assert_array_equal(serial, parallel)
    if ppv_only:
        np.testing.assert_array_equal(parallel, expected)
    else:
        np.testing.assert_array_equal(parallel[:, ::2], expected[:, ::2])
        # The original ROCKET kernel uses fastmath; maximum values can differ
        # by floating-point rounding across compilers without changing PPV.
        np.testing.assert_allclose(parallel[:, 1::2], expected[:, 1::2], rtol=1e-5, atol=1e-5)


def test_constant_series_and_output_ownership(original_transform):
    X = np.full((9, 32), 4.25, dtype=np.float64)
    seeds = [0, 1, 2, 3, 4]
    expected = original_transform["transform_seeds"](X, seeds, X.shape[1])
    first = transform_features(X, seeds, threads=4)
    np.testing.assert_array_equal(first, expected)
    first[:] = -10
    second = transform_features(X, seeds, threads=4)
    np.testing.assert_array_equal(second, expected)
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(X, np.full_like(X, 4.25))


def test_kernel_generation_does_not_depend_on_call_order(original_transform):
    X = np.random.default_rng(14).normal(size=(7, 41))
    expected = original_transform["transform_seeds"](X, [91, 2, 0], X.shape[1])
    transform_features(X, [5, 6, 7], threads=4)
    np.testing.assert_array_equal(transform_features(X, [91, 2, 0], threads=4), expected)
    np.testing.assert_array_equal(transform_features(X, [0, 2, 91], threads=1), expected[:, ::-1])


@pytest.mark.parametrize("matrix_kind", ["positive_definite", "indefinite", "nonsymmetric"])
def test_lambda_matches_original_and_does_not_mutate_inputs(matrix_kind):
    rng = np.random.default_rng(348)
    base = rng.normal(size=(8, 8))
    if matrix_kind == "positive_definite":
        A = base.T @ base + np.eye(8)
    elif matrix_kind == "indefinite":
        Q, _ = np.linalg.qr(base)
        A = Q @ np.diag([-5, -3, -2, 2, 3, 4, 5, 7]) @ Q.T
    else:
        A = base + 4 * np.eye(8)
    b = rng.normal(size=(8, 3))
    X = rng.normal(size=(37, 8))
    y = rng.integers(0, 3, size=37)
    lambdas = np.array([0.5, 0.001, 1.0, 0.04, 0.2])
    inputs = [A, b, X, y, lambdas]
    snapshots = [item.copy() for item in inputs]
    expected = reference_lambda(A, b, X, y, lambdas)
    for threads in (1, 4):
        check_lambda_result(select_lambda(A, b, X, y, lambdas, threads=threads), expected)
    for item, snapshot in zip(inputs, snapshots):
        np.testing.assert_array_equal(item, snapshot)


@pytest.mark.parametrize("correct_class", [0, 1])
def test_ties_choose_first_class_and_largest_lambda(correct_class):
    A = np.eye(3)
    b = np.ones((3, 2))
    X = np.zeros((7, 3))
    y = np.full(7, correct_class)
    lambdas = np.array([0.2, 1.0, 0.01, 1.0, 0.4])
    for threads in (1, 4):
        lam, accuracy, scores = select_lambda(A, b, X, y, lambdas, threads=threads)
        assert lam == 1.0
        assert accuracy == float(correct_class == 0)
        np.testing.assert_array_equal(scores, np.full(lambdas.size, accuracy))


def test_singular_system_is_reported_after_parallel_region():
    A = -np.eye(3)
    b = np.ones((3, 2))
    X = np.ones((4, 3))
    y = np.zeros(4, dtype=np.int64)
    for threads in (1, 4):
        with pytest.raises(np.linalg.LinAlgError):
            select_lambda(A, b, X, y, [0.1, 1.0, 2.0], threads=threads)


@pytest.mark.parametrize("lambdas", [[], [0.0], [-0.1], [np.nan], [np.inf], [[0.1, 1.0]]])
def test_invalid_lambda_grid(lambdas):
    with pytest.raises(ValueError):
        select_lambda(np.eye(3), np.ones((3, 2)), np.ones((4, 3)),
                      np.zeros(4, dtype=np.int64), lambdas, threads=2)


@pytest.mark.parametrize(
    "A,b,X,y",
    [
        (np.ones((3, 2)), np.ones((3, 2)), np.ones((4, 3)), np.zeros(4)),
        (np.eye(3), np.ones((2, 2)), np.ones((4, 3)), np.zeros(4)),
        (np.eye(3), np.ones((3, 2)), np.ones((4, 2)), np.zeros(4)),
        (np.eye(3), np.ones((3, 2)), np.ones((4, 3)), np.zeros(3)),
        (np.eye(3), np.ones((3, 2)), np.empty((0, 3)), np.empty(0)),
        (np.eye(3), np.empty((3, 0)), np.ones((4, 3)), np.zeros(4)),
    ],
)
def test_incompatible_shapes(A, b, X, y):
    with pytest.raises(ValueError):
        select_lambda(A, b, X, y.astype(np.int64), [0.1, 1.0], threads=2)


def test_concurrent_calls_keep_native_workspaces_private():
    """Independent Python callers may overlap after the binding releases GIL."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    rng = np.random.default_rng(931)
    seeds = list(range(24))
    lambdas = np.logspace(-3, 0, 9)
    jobs, expected = [], []
    for _ in range(3):
        series = rng.normal(size=(64, 96))
        factor = rng.normal(size=(24, 24))
        A = factor.T @ factor + np.eye(24)
        b = rng.normal(size=(24, 3))
        X = rng.normal(size=(128, 24))
        y = rng.integers(0, 3, size=128)
        jobs.append((series, A, b, X, y))
        # Warm the deterministic kernel-parameter cache before the pool starts;
        # the regression then stresses concurrent native calls, not JIT setup.
        expected.append((transform_features(series, seeds, threads=1),
                         reference_lambda(A, b, X, y, lambdas)))

    ready = Barrier(len(jobs), timeout=20)

    def run(index):
        series, A, b, X, y = jobs[index]
        threads = 1 + index % 2
        ready.wait()
        features = transform_features(series, seeds, threads=threads)
        ready.wait()
        result = select_lambda(A, b, X, y, lambdas, threads=threads)
        return features, result

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(run, range(len(jobs))))
    for (features, result), (reference_features, reference_result) in zip(results, expected):
        np.testing.assert_array_equal(features, reference_features)
        check_lambda_result(result, reference_result)
    for i in range(len(results)):
        for j in range(i):
            assert not np.shares_memory(results[i][0], results[j][0])
            assert not np.shares_memory(results[i][1][2], results[j][1][2])


def test_blas_thread_limit_preserves_requested_openmp_team():
    """BLAS may share libomp and set its default to one during adapter import.

    Our outer team must still honor the original OMP_NUM_THREADS. A subprocess
    tests runtime initialization without depending on earlier tests' settings.
    """
    import json
    import os

    environment = dict(os.environ, OMP_NUM_THREADS="3", OMP_DYNAMIC="FALSE",
                       OMP_THREAD_LIMIT="3", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1")
    program = (
        "import json\n"
        "import sys\n"
        "sys.path.insert(0, 'code')\n"
        "import openmp_utils\n"
        "print(json.dumps(openmp_utils.native.runtime_info()))\n"
    )
    result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=environment,
                            check=True, capture_output=True, text=True, timeout=30)
    info = json.loads(result.stdout)
    assert info["default_threads"] == 3
    assert info["probe_team_size"] == 3


def test_empty_and_invalid_seeds():
    X = np.random.default_rng(57).normal(size=(8, 32))
    for ppv_only in (False, True):
        empty = transform_features(X, [], ppv_only=ppv_only, threads=2)
        assert empty.shape == (len(X), 0)
        assert empty.dtype == np.float32
    for seeds in ([-1], [2**31]):
        with pytest.raises(ValueError):
            transform_features(X, seeds, threads=2)


def test_changed_input_never_reuses_stale_features(original_transform):
    rng = np.random.default_rng(14)
    X = rng.normal(size=(13, 41))
    seeds = [0, 2, 7, 9, 19, 31]
    first = transform_features(X, seeds, threads=2)
    X[:] = rng.normal(size=X.shape)
    expected = original_transform["transform_seeds"](X, seeds, X.shape[1])
    actual = transform_features(X, seeds, threads=2)
    assert np.any(first != expected)
    np.testing.assert_array_equal(actual, expected)


def test_finite_inputs_that_overflow_shift_are_reported():
    with pytest.raises(np.linalg.LinAlgError):
        select_lambda(np.eye(3) * 1e308, np.ones((3, 2)), np.ones((4, 3)),
                      np.zeros(4, dtype=np.int64), [0.1, 1e308], threads=2)


def test_native_rejects_unaligned_buffers_and_adapter_repairs_them():
    from openmp_utils import native

    def unaligned(shape, dtype, order="C"):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        array = np.ndarray(shape, dtype=dtype, buffer=bytearray(size + 1), offset=1, order=order)
        assert not array.flags.aligned
        return array

    X = unaligned((4, 16), np.float32)
    X[:] = np.arange(16, dtype=np.float32)
    assert X.flags.c_contiguous
    with pytest.raises(ValueError, match="aligned"):
        native.transform(X, np.ones((1, 11), dtype=np.float32), np.zeros(1, dtype=np.float32),
                         np.array([7], dtype=np.int32), np.ones(1, dtype=np.int32),
                         np.zeros(1, dtype=np.int32), True, 2)
    np.testing.assert_array_equal(transform_features(X, [0, 1], threads=2),
                                  transform_features(X.copy(), [0, 1], threads=1))

    A = unaligned((3, 3), np.float64, order="F")
    A[:] = np.eye(3)
    assert A.flags.f_contiguous
    b = np.asfortranarray(np.ones((3, 2)))
    features = np.ones((4, 3))
    y = np.zeros(4, dtype=np.int64)
    lambdas = np.array([0.1, 1.0])
    with pytest.raises(ValueError, match="aligned"):
        native.lambda_scores(A, b, features, y, lambdas, 2)
    check_lambda_result(select_lambda(A, b, features, y, lambdas, threads=2),
                        reference_lambda(A, b, features, y, lambdas))


@pytest.mark.parametrize("dataset", ["gunpoint", "italy_power_demand"])
def test_predictions_on_real_official_test_split(original_transform, dataset):
    """Small predictive regression; it does not claim statistical generalization.

    Only the official training split is used to choose lambda. Both versions
    refit on the full training split and evaluate the untouched official test
    split. Accuracy can be modest with 32 kernels; equality is the contract.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sktime import datasets

    loader = getattr(datasets, "load_" + dataset)
    X_train, y_train = loader(split="train", return_type="numpy3D")
    X_test, y_test = loader(split="test", return_type="numpy3D")
    X_train, X_test = X_train[:, 0, :], X_test[:, 0, :]
    encoder = LabelEncoder().fit(y_train)
    y_train, y_test = encoder.transform(y_train), encoder.transform(y_test)
    seeds = list(range(32))
    lambdas = np.logspace(-3, 0, 12)
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=0.25, random_state=42, stratify=y_train
    )
    train_original = original_transform["transform_seeds"](X_train, seeds, X_train.shape[1])
    test_original = original_transform["transform_seeds"](X_test, seeds, X_test.shape[1])
    targets = np.eye(len(encoder.classes_))[y_train]

    def training_system(features, indices):
        # Preserve the original float32 Gram-product arithmetic before its
        # promotion by the lambda shift, rather than silently changing dtype.
        return features[indices].T @ features[indices], features[indices].T @ targets[indices]

    A, b = training_system(train_original, train_idx)
    expected = reference_lambda(A, b, train_original[val_idx], y_train[val_idx], lambdas)

    def predict(features, test_features, lam):
        A_full = features.T @ features
        b_full = features.T @ targets
        W = np.linalg.solve(A_full + lam * np.eye(A_full.shape[0]), b_full)
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-10
        return np.argmax(test_features @ W, axis=1)

    predicted_original = predict(train_original, test_original, expected[0])
    accuracy_original = np.mean(predicted_original == y_test)
    for threads in (1, 4):
        train_parallel = transform_features(X_train, seeds, threads=threads)
        test_parallel = transform_features(X_test, seeds, threads=threads)
        np.testing.assert_array_equal(train_parallel, train_original)
        np.testing.assert_array_equal(test_parallel, test_original)
        selected = select_lambda(
            A, b, train_parallel[val_idx], y_train[val_idx], lambdas, threads=threads
        )
        check_lambda_result(selected, expected)
        predicted_parallel = predict(train_parallel, test_parallel, selected[0])
        np.testing.assert_array_equal(predicted_parallel, predicted_original)
        assert np.mean(predicted_parallel == y_test) == accuracy_original


def test_full_federated_protocol_matches_original(original_transform, capsys):
    """Run untouched original/main and current/main without CLI or network.

    ArrowHead is a real, bundled three-class series dataset. Three classes avoid
    the unrelated historical binary coef_ shape assumption in the baseline
    under modern sklearn; neither the original AST nor estimator is patched.
    """
    import os
    import random
    from types import SimpleNamespace

    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sktime.datasets import load_arrow_head
    from utils import preprocess_data, split_function

    train, train_labels = load_arrow_head(split="train", return_type="numpy3D")
    test, test_labels = load_arrow_head(split="test", return_type="numpy3D")
    encoder = LabelEncoder().fit(train_labels)
    dataset = (train[:, 0, :], encoder.transform(train_labels),
               test[:, 0, :], encoder.transform(test_labels))
    baseline_source = subprocess.run(
        ["git", "show", "209009c55b6b4d84b748ac5043ab48ee568fe4e5:code/fed2rc_no_fluke.py"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    current_source = (ROOT / "code" / "fed2rc_no_fluke.py").read_text()

    def run(source, threads):
        trace = {"seeds": [], "predictions": [], "metrics": []}

        def transform(X, seeds, ts_length):
            trace["seeds"].append(np.asarray(seeds).copy())
            if threads is None:
                return original_transform["transform_seeds"](X, seeds, ts_length)
            return transform_features(X, seeds, threads=threads)

        def accuracy(y, predictions):
            trace["predictions"].append(np.asarray(predictions).copy())
            return accuracy_score(y, predictions)

        offline_wandb = SimpleNamespace(
            init=lambda **kwargs: None, finish=lambda: None,
            Api=lambda: SimpleNamespace(runs=lambda project: []),
            log=lambda metrics: trace["metrics"].append(dict(metrics)),
        )
        namespace = {
            "np": np, "os": os, "random": random, "wandb": offline_wandb,
            "args": SimpleNamespace(debug=True, validation=False, datasets=["ArrowHead"]),
            "n_clients": 2, "n_kernels": 32, "n_rounds": 1,
            "top_k_per_client": 16, "EXPERIMENT_SEEDS": [1],
            "get_binary_dataset_names": lambda: [],
            "get_three_classes_dataset_names": lambda: ["ArrowHead"],
            "get_four_classes_dataset_names": lambda: [],
            "get_multiclasses_dataset_names": lambda: [],
            "load_dataset": lambda name: tuple(array.copy() for array in dataset),
            "preprocess_data": preprocess_data, "split_function": split_function,
            "RocketKernel": original_transform["RocketKernel"],
            "RidgeClassifierCV": RidgeClassifierCV, "train_test_split": train_test_split,
            "transform_seeds": transform, "accuracy_score": accuracy, "f1_score": f1_score,
            "select_lambda": lambda *args: select_lambda(*args, threads=threads),
        }
        main = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        exec(compile(ast.Module(body=[main], type_ignores=[]), "protocol_main.py", "exec"), namespace)
        namespace["main"]()
        return trace

    expected = run(baseline_source, None)
    assert len(expected["metrics"]) == 1
    assert len(expected["predictions"][-1]) == len(test_labels)
    for threads in (1, 4):
        actual = run(current_source, threads)
        assert len(actual["seeds"]) == len(expected["seeds"])
        for seeds, reference_seeds in zip(actual["seeds"], expected["seeds"]):
            np.testing.assert_array_equal(seeds, reference_seeds)
        np.testing.assert_array_equal(actual["predictions"][-1], expected["predictions"][-1])
        assert actual["metrics"] == expected["metrics"]
    capsys.readouterr()
