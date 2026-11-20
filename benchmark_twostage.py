"""Small standalone comparison of the original LU and two-stage lambda search."""
import argparse
import os
from statistics import median
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from openmp_utils import select_lambda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, nargs="+", default=[128, 900, 1024])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lambdas", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(*args.features, args.threads, args.lambdas, args.repeats) < 1:
        parser.error("all sizes, threads and repeats must be positive")
    lambdas = np.logspace(-3, 0, args.lambdas)
    print("Full lambda search, prepared float64 inputs; BLAS=1; warmup excluded.")
    print("D | lambdas | threads | LU ms | two-stage ms | speedup")
    for d in args.features:
        rng = np.random.default_rng(2026+d)
        z = rng.uniform(0, 1, size=(256, d))
        A = np.asfortranarray(z.T@z)
        b = np.asfortranarray(z.T@np.eye(3)[rng.integers(3, size=256)])
        X, y = rng.uniform(0, 1, size=(64, d)), rng.integers(3, size=64)
        def call(solver):
            return select_lambda(A, b, X, y, lambdas, threads=args.threads, solver=solver)
        expected = call("lu")
        def check(result):
            assert result[:2] == expected[:2]
            np.testing.assert_array_equal(result[2], expected[2])
        check(call("two_stage"))  # Warmup, including optional import.
        times = {"lu": [], "two_stage": []}
        for _ in range(args.repeats):
            for solver in rng.permutation(["lu", "two_stage"]):
                start = perf_counter()
                result = call(solver)
                times[solver].append(perf_counter()-start)
                check(result)  # Outside timed region.
        lu, two = median(times["lu"]), median(times["two_stage"])
        print(f"{d} | {args.lambdas} | {args.threads} | {lu*1000:.3f} | {two*1000:.3f} | {lu/two:.2f}x", flush=True)


if __name__ == "__main__":
    main()
