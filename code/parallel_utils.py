"""Native thread maps for the univariate Fed2RC pipeline.

Set NUMBA_NUM_THREADS before starting Python. For an OpenMP runtime, also set
NUMBA_THREADING_LAYER=omp (requires a Numba build with OpenMP support).
On macOS with Accelerate BLAS, set VECLIB_MAXIMUM_THREADS=1 before starting
Python: threadpoolctl cannot control or inspect that backend's thread count.
The ROCKET adapter uses sktime's private numerical kernel; regression tests
against Rocket.transform are required when updating sktime.
"""

from collections import OrderedDict
from hashlib import sha256
from threading import Lock

import numpy as np
import scipy.linalg  # Load the LAPACK/BLAS runtime before limiting its threads.
from numba import config, njit, prange, threading_layer

# Workqueue cannot serve simultaneous Python callers. Choose a threadsafe native
# backend by default; NUMBA_THREADING_LAYER=omp explicitly requests OpenMP.
if config.THREADING_LAYER == "default":
    config.THREADING_LAYER = "threadsafe"
try:
    _ACTIVE_LAYER = threading_layer()
except ValueError:  # No parallel kernel has initialized a runtime yet.
    _ACTIVE_LAYER = config.THREADING_LAYER
if _ACTIVE_LAYER == "workqueue":
    raise RuntimeError("Fed2RC requires a threadsafe Numba backend: use omp or tbb")

from sktime.transformations.panel.rocket import Rocket
from sktime.transformations.panel.rocket._rocket_numba import _apply_kernel_univariate
from threadpoolctl import threadpool_limits

# One process-wide policy, established once before any parallel work. Repeated
# per-call contexts would race when independent callers restore global limits.
# Parallelism belongs to the outer maps: request one BLAS thread per solve.
# This covers supported runtimes; Accelerate needs the startup setting above.
# Applications must not change this policy while a map is running.
_BLAS_LIMIT = threadpool_limits(limits=1, user_api="blas")


class FeatureCache:
    """Bounded LRU of complete, immutable feature columns.

    Payload <= max_bytes, entries <= max_entries. Python metadata and active
    caller references are additional: eviction never invalidates a reader.
    No lock is held during hashing, ROCKET preparation or numerical computation.
    Concurrent misses may duplicate computation; publication is atomic.
    """

    def __init__(self, max_bytes=128 * 1024**2, max_entries=8192):
        if max_bytes < 0 or max_entries < 0:
            raise ValueError("Cache limits must be nonnegative")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = Lock()

    @property
    def nbytes(self):
        with self._lock:
            return self._bytes

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def get_many(self, keys):
        # Snapshot references under one lock, then consume them without a lock.
        with self._lock:
            values = []
            for key in keys:
                value = self._entries.get(key)
                if value is not None:
                    self._entries.move_to_end(key)
                values.append(value)
            return values

    def put(self, key, value):
        if value.nbytes > self.max_bytes or self.max_entries == 0:
            return
        # A bytes-backed array cannot be made writable by another reader.
        # Copy outside the critical section; never retain a whole batch through
        # a small view, which would defeat payload accounting and eviction.
        value = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
        with self._lock:
            if key in self._entries:
                return
            while self._entries and (
                self._bytes + value.nbytes > self.max_bytes
                or len(self._entries) >= self.max_entries
            ):
                _, old = self._entries.popitem(last=False)
                self._bytes -= old.nbytes
            self._entries[key] = value
            self._bytes += value.nbytes


FEATURE_CACHE = FeatureCache()


@njit(parallel=True, nogil=True, cache=True)
def _transform_map(X, weights, lengths, biases, dilations, paddings, out):
    # Ownership: iteration s writes only out[s, :, :]. Seed-major storage keeps
    # each task's writes contiguous; only block boundaries can share cache lines.
    # The existing scalar sktime kernel preserves ROCKET's arithmetic (including
    # its own fastmath setting). No Rocket/Pandas/Python objects enter prange.
    for s in prange(len(lengths)):
        for row in range(X.shape[0]):
            ppv, maximum = _apply_kernel_univariate(
                X[row], weights[s, :lengths[s]], lengths[s], biases[s],
                dilations[s], paddings[s],
            )
            out[s, row, 0] = ppv
            if out.shape[2] == 2:
                out[s, row, 1] = maximum


def transform_features(X, seeds, *, ppv_only=True, caching=True, cache=FEATURE_CACHE):
    """Return float32 features in requested seed order, including duplicates.

    Inputs must not be mutated concurrently while this call snapshots them.
    Cache identity includes raw contents, shape, dtype and feature mode, so it
    cannot confuse training/validation data or reused client identifiers.
    Hashing costs O(N*T); cached PPV storage costs 4*N*S bytes (twice for both
    features). For N=10,000, S=1,000 the PPV payload is 38.15 MiB. The returned
    matrix, snapshot/normalization and miss workspace are additional O(N*S+N*T).
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] < 2:
        raise ValueError("Expected a nonempty univariate panel (samples, timepoints >= 2)")
    if X.dtype.kind not in "fiu" or not np.isfinite(X).all():
        raise ValueError("ROCKET requires finite real input")
    X = np.array(X, order="C", copy=True)
    seeds = [int(seed) for seed in seeds]
    if any(seed < 0 or seed > 2**31 - 1 for seed in seeds):
        raise ValueError("ROCKET seeds must fit a nonnegative int32")
    width = 1 if ppv_only else 2
    if not seeds:
        return np.empty((len(X), 0), dtype=np.float32)
    unique = list(dict.fromkeys(seeds))
    if caching:
        identity = (X.shape, X.dtype.str, sha256(memoryview(X).cast("B")).digest(), ppv_only)
        keys = [(identity, seed) for seed in unique]
        columns = cache.get_many(keys)
    else:
        keys = [None] * len(unique)
        columns = [None] * len(unique)
    missing = [i for i, column in enumerate(columns) if column is None]
    if missing:
        # Match Rocket._transform: normalize in the input precision, THEN cast.
        Xn = np.ascontiguousarray(
            (X - X.mean(axis=1, keepdims=True))
            / (X.std(axis=1, keepdims=True) + 1e-8), dtype=np.float32,
        )
        weights = np.zeros((len(missing), 11), dtype=np.float32)
        lengths = np.empty(len(missing), dtype=np.int32)
        biases = np.empty(len(missing), dtype=np.float32)
        dilations = np.empty(len(missing), dtype=np.int32)
        paddings = np.empty(len(missing), dtype=np.int32)
        dummy = np.zeros((1, 1, X.shape[1]))
        # Sequential preparation preserves sktime's per-seed RNG semantics.
        for j, i in enumerate(missing):
            rocket = Rocket(num_kernels=1, random_state=unique[i], n_jobs=1).fit(dummy)
            w, length, bias, dilation, padding, _, _ = rocket.kernels
            weights[j, :length[0]] = w
            lengths[j], biases[j] = length[0], bias[0]
            dilations[j], paddings[j] = dilation[0], padding[0]
        out = np.empty((len(missing), len(X), width), dtype=np.float32)
        _transform_map(Xn, weights, lengths, biases, dilations, paddings, out)
        # Return from prange is the barrier. Only complete results are published.
        for j, i in enumerate(missing):
            columns[i] = out[j]
            if caching:
                cache.put(keys[i], out[j])
    positions = {seed: i for i, seed in enumerate(unique)}
    return np.concatenate([columns[positions[seed]] for seed in seeds], axis=1)


@njit(cache=True, nogil=True)
def _lambda_accuracy(A, b, X, y, lam):
    # Each task owns its shifted matrix, LU workspace, weights and predictions.
    shifted = A.copy()
    for j in range(A.shape[0]):
        shifted[j, j] += lam
    # Exceptions must not escape a native worker: Numba can otherwise return a
    # SystemError or leave output cells unwritten. Publish failure as data and
    # raise on the calling thread after the map's barrier.
    try:
        W = np.linalg.solve(shifted, b)
    except Exception:
        return np.nan
    for c in range(W.shape[1]):
        norm = 0.0
        for j in range(W.shape[0]):
            norm += W[j, c] * W[j, c]
        norm = np.sqrt(norm) + 1e-10
        if not np.isfinite(norm):
            return np.nan
        for j in range(W.shape[0]):
            W[j, c] /= norm
    predictions = X @ W
    correct = 0
    for row in range(len(y)):
        # Strict comparison retains NumPy argmax's first-class tie rule.
        best = 0
        for c in range(W.shape[1]):
            if not np.isfinite(predictions[row, c]):
                return np.nan
        for c in range(1, W.shape[1]):
            if predictions[row, c] > predictions[row, best]:
                best = c
        correct += best == y[row]
    return correct / len(y)


@njit(parallel=True, nogil=True, cache=True)
def _lambda_map(A, b, X, y, lambdas, accuracies):
    for i in prange(len(lambdas)):
        # One final store per O(d^3 + N*d*C) task; no contended best-so-far.
        # Adjacent cells may share lines, but are never repeatedly updated.
        accuracies[i] = _lambda_accuracy(A, b, X, y, lambdas[i])


def select_lambda(A, b, X, y, lambdas):
    """LAPACK LU map followed by a deterministic lexicographic reduction.

    O(L*d^3) work, O(P*(d^2+d*C+N*C)) solver workspace for P workers, O(L)
    scores. Inputs are read-only during the call. Singular systems raise the
    same LinAlgError as np.linalg.solve; positive ridge shifts are not assumed
    to make an arbitrary (e.g. compressed/subtracted) matrix positive definite.
    """
    A, b, X = [np.ascontiguousarray(a, dtype=np.float64) for a in (A, b, X)]
    y = np.asarray(y)
    lambdas = np.ascontiguousarray(lambdas, dtype=np.float64)
    if (A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1]
            or b.ndim != 2 or b.shape[0] != A.shape[0] or b.shape[1] == 0
            or X.ndim != 2 or X.shape[1] != A.shape[0]
            or y.ndim != 1 or len(y) != len(X) or len(y) == 0
            or lambdas.ndim != 1 or len(lambdas) == 0):
        raise ValueError("Incompatible or empty ridge search inputs")
    if not all(np.isfinite(a).all() for a in (A, b, X, y, lambdas)):
        raise ValueError("Ridge search inputs must be finite")
    if np.any(lambdas <= 0):
        raise ValueError("Ridge lambdas must be positive")
    accuracies = np.empty(len(lambdas), dtype=np.float64)
    _lambda_map(A, b, X, np.ascontiguousarray(y), lambdas, accuracies)
    failed = np.flatnonzero(~np.isfinite(accuracies))
    if len(failed):
        raise np.linalg.LinAlgError(
            f"Singular system or nonfinite result at lambda index {failed[0]}"
        )
    # Barrier has completed. O(L) reduction is negligible next to L solves.
    best = max(range(len(lambdas)), key=lambda i: (accuracies[i], lambdas[i]))
    return float(lambdas[best]), float(accuracies[best]), accuracies
