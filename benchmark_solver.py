"""Matched native-code benchmark; no BLAS, sktime, dataset downloads or I/O timed.

Run: NUMBA_NUM_THREADS=4 python benchmark_solver.py 96 100 4 5
Compare: ./benchmark_solver 96 100 4 5
Arguments: matrix dimension, lambda count, threads, repetitions.
Both programs time the same dense LU with partial pivoting, triangular solves,
weight normalization, classification, per-lambda allocations and final reduce.
Generation, JIT compilation, warmup, residual checks and JSON output are excluded.
Compare on the same idle machine with the same CPU allocation/affinity. Neither
program uses fastmath; C++ must be built with -ffp-contract=off, without -Ofast.
Numba's reported threading layer need not be OpenMP on every supported platform.
"""

import argparse
import json
import math
import statistics
from time import perf_counter

import numpy as np
from numba import config, njit, prange, set_num_threads, threading_layer


def make_problem(n, count):
    # Integer modular formulas avoid language-specific RNG streams. A is dense,
    # symmetric and strictly diagonally dominant with positive diagonal => SPD.
    A = np.empty((n, n), dtype=np.float64)
    b = np.empty((n, 3), dtype=np.float64)
    X = np.empty((2 * n, n), dtype=np.float64)
    y = np.arange(2 * n, dtype=np.int64) % 3
    for i in range(n):
        for j in range(n):
            A[i, j] = 2.0 + i / n if i == j else ((i + j) % 7 - 3) / (20.0 * n)
        for c in range(3):
            b[i, c] = ((i * 3 + c * 5) % 17 - 8) / 8.0
    for r in range(2 * n):
        for j in range(n):
            X[r, j] = ((r * 7 + j * 3) % 19 - 9) / 9.0
    lambdas = np.array([math.pow(10.0, -3.0 + 3.0 * i / (count - 1))
                        for i in range(count)], dtype=np.float64)
    return A, b, X, y, lambdas


@njit(nogil=True, cache=True)
def lu_solve(A, b, lam):
    """Scalar float64 LU/partial pivoting, mirrored line by line in C++."""
    n, classes = b.shape
    lu = A.copy()
    W = b.copy()
    for i in range(n):
        lu[i, i] += lam
    for k in range(n):
        pivot = k
        for i in range(k + 1, n):
            if abs(lu[i, k]) > abs(lu[pivot, k]):
                pivot = i
        if lu[pivot, k] == 0.0:
            raise np.linalg.LinAlgError("Singular matrix")
        if pivot != k:
            for j in range(n):
                tmp = lu[k, j]
                lu[k, j] = lu[pivot, j]
                lu[pivot, j] = tmp
            for c in range(classes):
                tmp = W[k, c]
                W[k, c] = W[pivot, c]
                W[pivot, c] = tmp
        for i in range(k + 1, n):
            lu[i, k] /= lu[k, k]
            for j in range(k + 1, n):
                lu[i, j] -= lu[i, k] * lu[k, j]
    for i in range(n):
        for c in range(classes):
            for j in range(i):
                W[i, c] -= lu[i, j] * W[j, c]
    for i in range(n - 1, -1, -1):
        for c in range(classes):
            for j in range(i + 1, n):
                W[i, c] -= lu[i, j] * W[j, c]
            W[i, c] /= lu[i, i]
    return W


@njit(nogil=True, cache=True)
def accuracy(A, b, X, y, lam):
    W = lu_solve(A, b, lam)
    for c in range(b.shape[1]):
        norm = 0.0
        for j in range(len(A)):
            norm += W[j, c] * W[j, c]
        norm = math.sqrt(norm) + 1e-10
        for j in range(len(A)):
            W[j, c] /= norm
    correct = 0
    for row in range(len(X)):
        best_class = 0
        best_score = -math.inf
        for c in range(b.shape[1]):
            score = 0.0
            for j in range(len(A)):
                score += X[row, j] * W[j, c]
            if score > best_score:
                best_class, best_score = c, score
        correct += best_class == y[row]
    return correct / len(y)


@njit(parallel=True, nogil=True, cache=True)
def search(A, b, X, y, lambdas):
    scores = np.empty(len(lambdas), dtype=np.float64)
    for i in prange(len(lambdas)):
        # Read-only shared inputs; private LU/W; one store per expensive task.
        scores[i] = accuracy(A, b, X, y, lambdas[i])
    # Implicit join before reduction; largest lambda wins accuracy ties.
    best = 0
    for i in range(1, len(lambdas)):
        if (scores[i] > scores[best]
                or (scores[i] == scores[best] and lambdas[i] > lambdas[best])):
            best = i
    return scores, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("n", 96), ("count", 100), ("threads", 4), ("repeats", 5)):
        parser.add_argument(name, nargs="?", type=int, default=default)
    args = parser.parse_args()
    if (args.n < 1 or args.count < 2 or args.repeats < 1
            or not 1 <= args.threads <= config.NUMBA_NUM_THREADS):
        parser.error("Require n>=1, count>=2, repeats>=1 and an available thread count")
    problem = make_problem(args.n, args.count)
    times = [[], []]
    results = []
    for threads in (1, args.threads):
        set_num_threads(threads)
        results.append(search(*problem))  # Compile AND warm both thread counts.
    for repetition in range(args.repeats):
        for mode in ((0, 1) if repetition % 2 == 0 else (1, 0)):
            set_num_threads((1, args.threads)[mode])
            start = perf_counter()
            result = search(*problem)
            times[mode].append(perf_counter() - start)
            if result[1] != results[0][1] or not np.array_equal(result[0], results[0][0]):
                raise RuntimeError("Serial/parallel results differ")
            results[mode] = result
    A, b, _, _, lambdas = problem
    scores, best = results[1]
    W = lu_solve(A, b, lambdas[best])
    shifted = A + lambdas[best] * np.eye(args.n)
    residual = np.linalg.norm(shifted @ W - b) / (
        np.linalg.norm(shifted) * np.linalg.norm(W) + np.linalg.norm(b))
    if not np.isfinite(residual) or residual > 1e-12:
        raise RuntimeError("LU residual exceeds tolerance")
    serial, parallel = map(statistics.median, times)
    print(json.dumps(dict(
        backend=threading_layer(), n=args.n, count=args.count, threads=args.threads,
        repeats=args.repeats, serial_seconds=serial, parallel_seconds=parallel,
        speedup=serial / parallel, best_index=int(best), best_lambda=float(lambdas[best]),
        accuracy=float(scores[best]), residual=float(residual),
        scores=scores.tolist(), solution=W.ravel().tolist(),
    )))


if __name__ == "__main__":
    main()
