"""Optional local timings; native strong scaling is distinct from baseline gains."""
import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter

# Set before importing numerical libraries; outer OpenMP owns the parallelism.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

import numpy as np
from sktime.transformations.panel.rocket import Rocket
from threadpoolctl import threadpool_info
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent
BASELINE = "209009c55b6b4d84b748ac5043ab48ee568fe4e5"
sys.path.insert(0, str(ROOT / "code"))
from openmp_utils import native, select_lambda, transform_features


def original_transform():
    """Load only the two original definitions, avoiding app imports/side effects."""
    source = subprocess.run(["git", "show", f"{BASELINE}:code/utils.py"],
                            cwd=ROOT, check=True, capture_output=True, text=True).stdout
    definitions = [node for node in ast.parse(source).body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                   and node.name in {"RocketKernel", "transform_seeds"}]
    namespace = {"np": np, "Rocket": Rocket}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "baseline", "exec"), namespace)
    return namespace["transform_seeds"]


def original_lambda(A, b, X, y, lambdas):
    """The baseline LU, column normalization, prediction and tie policy."""
    accuracies = np.empty(len(lambdas))
    for i, lam in enumerate(lambdas):
        W = np.linalg.solve(A + lam * np.eye(A.shape[0]), b)
        W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-10)
        accuracies[i] = accuracy_score(y, np.argmax(X @ W, axis=1))
    best = max(range(len(lambdas)), key=lambda i: (accuracies[i], lambdas[i]))
    return float(lambdas[best]), float(accuracies[best]), accuracies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(p < 1 for p in args.threads):
        parser.error("repeats and thread counts must be positive")
    threads = sorted(set([1, *args.threads]))  # 1 is required for native speedup.
    rng = np.random.default_rng(381)
    N, T, K, d, C, L = 256, 256, 128, 128, 3, 64
    series, seeds = rng.normal(size=(N, T)), list(range(K))
    design = rng.normal(size=(N, d))
    A = design.T @ design + np.eye(d)
    b = rng.normal(size=(d, C))
    validation, y = rng.normal(size=(N, d)), rng.integers(C, size=N)
    lambdas = np.logspace(-3, 0, L)
    baseline = original_transform()
    calls = {
        "transform/original": lambda: baseline(series, seeds, T),
        "lambda/original": lambda: original_lambda(A, b, validation, y, lambdas),
    }
    for p in threads:
        calls[f"transform/native_{p}"] = lambda p=p: transform_features(series, seeds, threads=p)
        calls[f"lambda/native_{p}"] = lambda p=p: select_lambda(A, b, validation, y, lambdas, threads=p)

    # These untimed calls compile sktime's reference kernels, fill the parameter
    # cache and warm every native team size. Stop before timing on any mismatch.
    expected_features, expected_lambda = calls["transform/original"](), calls["lambda/original"]()
    for p in threads:
        np.testing.assert_array_equal(calls[f"transform/native_{p}"](), expected_features)
        result = calls[f"lambda/native_{p}"]()
        if result[:2] != expected_lambda[:2]:
            raise AssertionError("lambda or best accuracy differs")
        np.testing.assert_array_equal(result[2], expected_lambda[2])

    raw = {name: [] for name in calls}
    order_rng = random.Random(42)
    orders = []
    for _ in range(args.repeats):
        order = list(calls)
        order_rng.shuffle(order)
        orders.append(order)
        for name in order:
            start = perf_counter()
            calls[name]()
            raw[name].append(perf_counter() - start)
    medians = {name: float(np.median(times)) for name, times in raw.items()}
    results = {}
    for kernel in ("transform", "lambda"):
        results[kernel] = {
            "original": {"median_seconds": medians[f"{kernel}/original"],
                         "seconds": raw[f"{kernel}/original"]},
            "native": [{"threads": p, "median_seconds": medians[f"{kernel}/native_{p}"],
                        "seconds": raw[f"{kernel}/native_{p}"],
                        "native_speedup": medians[f"{kernel}/native_1"] / medians[f"{kernel}/native_{p}"],
                        "original_over_native": medians[f"{kernel}/original"] / medians[f"{kernel}/native_{p}"]}
                       for p in threads],
        }
    report = {
        "baseline_commit": BASELINE, "system": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(), "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "sktime", "numba", "threadpoolctl")},
        "runtime": native.runtime_info(), "threadpools": threadpool_info(),
        "dimensions": {"samples": N, "timepoints": T, "kernels": K, "features": d, "classes": C, "lambdas": L},
        "repeats": args.repeats, "validation": "exact PPV, lambda and accuracy-vector agreement",
        "timing_scope": "Full helper calls, including checks/packing; excludes imports/data/warmup. "
                        "Native parameter cache warm; original includes fit each call. BLAS single-threaded. "
                        "Native speedup compares identical code. Original ratio also includes serial optimizations and BLAS differences.",
        "execution_orders": orders, "results": results,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
