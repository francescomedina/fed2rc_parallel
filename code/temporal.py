"""Backend-independent validation and conservative temporal granularity policy.

The policy never changes numerical operations within one convolution output.
Explicit chunk counts are experiments; only ``auto`` avoids unprofitable splits.
"""

from numbers import Integral

import numpy as np


# Enough independent outputs to amortize scheduling and the feature reduction.
# Calibrated by benchmark_temporal.py; see TEMPORAL.md for measurements/limits.
MIN_OUTPUTS_PER_CHUNK = 4096
MIN_TAP_OPERATIONS_PER_WORKER = 32768
MIN_BATCH_OUTPUTS_PER_CHUNK = 64
MAX_AUTO_CHUNKS = 4


def validate_temporal_chunks(value):
    if isinstance(value, str) and value == "auto":
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value > 0:
        return int(value)
    raise ValueError("temporal_chunks must be a positive integer or 'auto'")


def resolve_temporal_chunks(value, timepoints, lengths, dilations, paddings,
                            threads, kernel_jobs, samples=1):
    """Choose a single chunk count; short outputs never create empty tasks.

    Existing kernel parallelism remains the preferred path. Split only when
    that path has fewer jobs than threads and enough useful work per worker.
    The caller supplies its existing job count (Numba kernels / OpenMP tiles).
    """
    value = validate_temporal_chunks(value)
    if value == 1 or len(lengths) == 0:
        return 1
    # The common saturated/single-thread case needs no geometry arrays or scan.
    if value == "auto" and (threads <= 1 or kernel_jobs >= threads):
        return 1
    outputs = timepoints + 2 * np.asarray(paddings, dtype=np.int64) - (
        np.asarray(lengths, dtype=np.int64) - 1
    ) * np.asarray(dilations, dtype=np.int64)
    minimum = int(outputs.min())
    if minimum < 1:
        raise ValueError("invalid kernel geometry: convolution must have an output")
    if value != "auto":
        return min(value, minimum)
    pairs = samples * len(lengths)
    work_per_worker = samples * float(np.sum(outputs * np.asarray(lengths, dtype=np.int64))) / threads
    if work_per_worker < MIN_TAP_OPERATIONS_PER_WORKER:
        return 1
    # Entering the combined path also exposes sample/kernel pairs hidden inside
    # original kernel jobs/tiles. Many such pairs amortize shorter time chunks.
    # At least two chunks select this path; no original map is replaced.
    target = max(2, min(MAX_AUTO_CHUNKS, (threads + pairs - 1) // pairs))
    minimum_chunk = (MIN_OUTPUTS_PER_CHUNK if pairs < threads
                     else MIN_BATCH_OUTPUTS_PER_CHUNK)
    chunks = 1
    while chunks < target:
        candidate = chunks * 2
        if minimum // candidate < minimum_chunk:
            break
        chunks = candidate
    return chunks
