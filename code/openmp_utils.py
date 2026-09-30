"""Small Python adapter; all parallel loops live in native/*.hpp.

Inputs must not be mutated concurrently during a call. There is no global cache
of dataset features: that would require hashing/copying datasets or an explicit
immutable-dataset lifetime. Only small, data-independent kernel parameters are
memoized. Numba is used internally by sktime.fit, not for our parallel execution.
"""
from functools import lru_cache

import numpy as np
from sktime.transformations.panel.rocket import Rocket
from threadpoolctl import threadpool_limits

import _fed2rc_openmp as native

# Establish one process-wide policy AFTER importing the native OpenBLAS library.
# Nested per-call contexts would race when concurrent callers restore limits.
# Users must not change BLAS threading while native calls are running.
_BLAS_LIMIT = threadpool_limits(limits=1, user_api="blas")


@lru_cache(maxsize=4096)
def _kernel(timepoints, seed):
    """Preserve sktime's RNG; memoize immutable small parameters, never datasets."""
    rocket = Rocket(num_kernels=1, random_state=seed, n_jobs=1)
    rocket.fit(np.zeros((1, 1, timepoints)))
    weights, lengths, biases, dilations, paddings, _, _ = rocket.kernels
    # A tuple is immutable. lru_cache synchronizes metadata only; overlapping
    # misses may prepare the same deterministic kernel twice without corruption.
    return (tuple(float(w) for w in weights), int(lengths[0]), float(biases[0]),
            int(dilations[0]), int(paddings[0]))


def transform_features(X, seeds, *, ppv_only=True, caching=True, threads=0):
    """ROCKET features, preserving seed order; output is float32, column-major.

    Normalization costs O(N*T) once, not once per seed. No array of all
    convolution responses is materialized. `caching` controls kernel parameters
    only; every call reads the current X. `threads=0` uses the OpenMP environment.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] < 2:
        raise ValueError("Expected (samples > 0, timepoints >= 2)")
    if X.dtype.kind not in "fiu" or not np.isfinite(X).all():
        raise ValueError("ROCKET requires finite real input")
    seeds = [int(s) for s in seeds]
    if any(s < 0 or s > 2**31 - 1 for s in seeds):
        raise ValueError("Seeds must fit a nonnegative int32")
    # Same precision and epsilon as Rocket._transform; cast AFTER normalization.
    normalized = np.require(
        (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8),
        dtype=np.float32, requirements=["C", "A"],
    )
    k = len(seeds)
    weights = np.zeros((k, 11), dtype=np.float32)
    lengths = np.empty(k, dtype=np.int32)
    biases = np.empty(k, dtype=np.float32)
    dilations = np.empty(k, dtype=np.int32)
    paddings = np.empty(k, dtype=np.int32)
    prepare = _kernel if caching else _kernel.__wrapped__
    for i, seed in enumerate(seeds):
        w, length, bias, dilation, padding = prepare(X.shape[1], seed)
        weights[i, :length] = w
        lengths[i], biases[i], dilations[i], paddings[i] = length, bias, dilation, padding
    return native.transform(normalized, weights, biases, lengths, dilations, paddings,
                            ppv_only, threads)


def select_lambda(A, b, X, y, lambdas, *, threads=0):
    """General LU solves in parallel, then deterministic O(L) reduction.

    A/b are packed into LAPACK's column-major layout once; each worker reuses
    its private LU/RHS buffers for all assigned lambdas. X stays row-major for
    tiled BLAS prediction. Aligned compatible inputs require no adapter copy.
    """
    A, b = [np.require(a, dtype=np.float64, requirements=["F", "A"]) for a in (A, b)]
    X = np.require(X, dtype=np.float64, requirements=["C", "A"])
    y = np.asarray(y)
    if y.dtype.kind not in "iu" or y.ndim != 1:
        raise ValueError("y must be a vector of integer class indices")
    y = np.require(y, dtype=np.int64, requirements=["C", "A"])
    lambdas = np.require(lambdas, dtype=np.float64, requirements=["C", "A"])
    scores, status = native.lambda_scores(A, b, X, y, lambdas, threads)
    failed = np.flatnonzero(status)
    if len(failed):
        i = int(failed[0])
        raise np.linalg.LinAlgError(f"Lambda index {i}: solve/prediction failed (status {status[i]})")
    # The OpenMP barrier has completed: no shared best-so-far in the hot loop.
    best = max(range(len(lambdas)), key=lambda i: (scores[i], lambdas[i]))
    return float(lambdas[best]), float(scores[best]), scores
