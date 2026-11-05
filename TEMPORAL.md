# Temporal partitioning of ROCKET convolutions

Both implementations now support `temporal_chunks=1, 2, 4, "auto"`, as well as
other positive integer chunk counts. The default is **1**: the existing
kernel-only parallel implementation is retained. This is an additional
worksharing strategy, not nested parallelism.

- C++/OpenMP: `code/openmp_utils.py` -> `native/bindings.cpp` ->
  `rocket_map_temporal` in `native/rocket.hpp`.
- Numba: `code/parallel_utils.py`, restored from `numba-comparison` at
  `1f3ddba`. Its existing `_transform_map`, feature cache and lambda solver are
  retained; `_transform_temporal` is the new additional map.
- Shared validation/adaptive policy: `code/temporal.py`.

## Use

Build the extension after changing the C++ source:

```bash
source .venv/bin/activate
python setup.py build_ext --inplace
```

Set the thread limits before starting Python. Each numerical BLAS call remains
single-threaded; OpenMP/Numba owns the outer worksharing:

```bash
export OMP_NUM_THREADS=4
export OMP_DYNAMIC=FALSE
export NUMBA_NUM_THREADS=4
export NUMBA_THREADING_LAYER=omp
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

The existing public feature API gains an optional keyword argument:

```python
# From code/, or with PYTHONPATH=code:
from openmp_utils import transform_features as cpp_transform
from parallel_utils import transform_features as numba_transform

# X has shape (samples, timepoints); seeds select the same sktime ROCKET filters.
cpp_features = cpp_transform(X, seeds, ppv_only=False, threads=4,
                             temporal_chunks=2)
numba_features = numba_transform(X, seeds, ppv_only=False,
                                 temporal_chunks=2, caching=False)

# Conservative automatic selection for the current workload/thread count.
features = cpp_transform(X, seeds, temporal_chunks="auto")
```

Numba's thread count is controlled by `NUMBA_NUM_THREADS`/`numba.set_num_threads`.
The C++ helper retains its existing `threads` keyword. The Fluke transform
wrappers also accept `temporal_chunks` as a keyword.

The standalone driver supports both backends:

```bash
cd code
python fed2rc_no_fluke.py --engine parallel --temporal-chunks auto \
  --debug --datasets GunPoint --experiment-seeds 1 -c 2 -k 32 -r 1
python fed2rc_no_fluke.py --engine numba --temporal-chunks 2 \
  --debug --datasets GunPoint --experiment-seeds 1 -c 2 -k 32 -r 1
```

`--engine classic` keeps the original sktime/NumPy baseline and requires one
temporal chunk. `--engine parallel` still means native C++/OpenMP. Scheduling
is independent of the Numba feature cache: an all-hit call needs no convolution
at all. The benchmark disables application caches when comparing calculations.

## Partition global output positions

For input length T, kernel length L, dilation D and original padding P, the
number of convolution outputs is:

```
M = T + 2*P - (L-1)*D
```

Chunk c of C owns the half-open output interval:

```
first = floor(M*c/C)
stop  = floor(M*(c+1)/C)
```

For each output q in that interval, the original convolution starts at `q-P`.
Tap j reads global input index `q-P+j*D`, only when `0 <= index < T`.
This reproduces the original zero padding at the signal boundaries. The chunk
boundary adds **no padding** and does not restrict the input reads. Adjacent
tasks may read overlapping regions, which are read-only. No halo copy is made.

These intervals cover all M outputs exactly once, including odd M, dilation
crossing chunk boundaries, and kernels wider than short input series when
padding makes their geometry valid. Public helpers cap explicit chunk counts
at the shortest output domain. The low-level maps also correctly handle empty
intervals when a diagnostic test requests more chunks than outputs.

## Worksharing and reduction

The additional map flattens `(kernel, sample, temporal_chunk)` into a single
task index. Thus, a single kernel on a single series can occupy multiple CPU
workers, while other kernels/samples remain independent parallel tasks.

Numba uses one `prange` over map tasks followed by a second, **non-nested**
`prange` over sample/kernel pairs. C++ uses one OpenMP team with two `omp for`
loops and their implicit barrier. Each map task writes its own partial slot;
each reduction task owns its final feature cells. No locks or atomic updates
are needed for numerical output.

Each chunk produces an int64 positive count and a float32 maximum. Reduction
visits chunks in temporal order:

```
MAX = max(local_MAX)
PPV = float32(sum(local_positive_counts) / M)
```

We never average chunk PPVs: their interval sizes can differ. Only integer
counts are summed across tasks. The floating-point dot product of any individual
output retains its original tap order and arithmetic settings. Feature order
remains PPV, then MAX when requested, for each kernel in the requested seed order.

The Numba chunk kernel preserves sktime's `fastmath=True`; the C++ path retains
its existing strict arithmetic and `-ffp-contract=off`. Every temporal result
must equal its own original implementation exactly. Existing cross-backend
rounding differences are recorded separately, not attributed to partitioning.

## Memory

There are no arrays of all convolution responses and no per-task data-buffer
allocation. Scratch is allocated once per call before workers start:

```
int64 counts[N,K,C] + float32 maxima[N,K,C] = 12*N*K*C bytes
```

For N=1, K=1, C=4, this adds 48 bytes of array payload; for N=128, K=128, C=4,
it adds 768 KiB. Output allocation is unchanged: `4*N*K` bytes for PPV-only,
or `8*N*K` bytes for PPV and MAX. Allocator bookkeeping, runtime pools and
Numba's existing feature packing are additional. Scratch is independent of T.
The original path selected by chunks=1 allocates no temporal scratch.

C++ scratch is uninitialized on allocation: every worker writes its own slot
before the reduction barrier. This avoids a redundant serial zero-fill.

## Adaptive granularity

`auto` preserves the original map if one worker is requested or the existing
map already has at least as many jobs as workers. Existing jobs are kernels
for Numba and 32-sample-by-8-kernel tiles for C++.

For an underfilled map, the policy estimates total tap work from actual output
lengths, kernel lengths and sample count. It requires at least 32,768 estimated
tap iterations per worker. It then chooses two or four chunks:

- With fewer sample/kernel pairs than workers, require at least 4,096 outputs
  per chunk and aim to occupy the remaining workers.
- With enough pairs but insufficient original kernel/tile jobs, two chunks
  expose those independent pairs; require at least 64 outputs per chunk.
- Limit auto to four chunks, the largest split calibrated in the recorded
  sweep. Explicit N remains available for experiments on other machines.

This is a conservative cost heuristic, not a portable guarantee of the optimum.
It avoids splitting the short and already-saturated cases in our measurements.
The sample-aware rule also recovers underfilled C++ tiles with many sample/kernel
pairs, which a per-convolution-only length threshold misses. Gains in these
batched cases include finer sample/kernel scheduling; the one-series/one-kernel
case isolates genuine temporal parallelism.

## Correctness and actual parallel execution

```bash
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE NUMBA_NUM_THREADS=4 \
NUMBA_THREADING_LAYER=omp OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -m pytest -q
```

`tests/test_temporal.py` compares both paths to their untouched kernel-only maps
and Numba to the original sktime scalar kernel. It covers odd/even and short/long
signals, padding, dilation including zero, kernels wider than input, impulses
at partition seams, unequal partition sizes, empty intervals, arbitrary split
counts, random geometry, feature order/duplicate seeds, dtypes, strided inputs,
constant series, cache hits/misses, thread-count reproducibility, concurrent
callers and downstream classifier predictions.

Worker IDs are recorded from the **actual convolution tasks** for one series
and one kernel. Both backends must show distinct workers for the chunks. C++
accepts a private optional `_worker_ids` diagnostic buffer; Numba compiles a
diagnostic specialization with the same optional buffer. Normal calls omit it.

## Reproduce measurements

```bash
NUMBA_NUM_THREADS=4 NUMBA_THREADING_LAYER=omp OMP_NUM_THREADS=4 \
OMP_DYNAMIC=FALSE python benchmark_temporal.py \
  --threads 1 2 4 --repeats 9 --target-ms 15 --output results/temporal
```

The benchmark exercises nine shapes, original/2/4/auto strategies, both
backends and three CPU thread counts. All inputs/kernels are identical within
each comparison. Every configuration is compiled and warmed before randomized,
batched timing; correctness checks happen outside the timer. Medians, quartiles,
raw repetitions, call counts, execution orders, source hashes, versions, runtime
information and actual worker IDs are saved.

- **Prepared scope:** pre-normalized float32 data and pre-generated kernels.
  Includes allocation and existing API/feature-packing overhead. C++ includes
  its native input validation; the internal Numba map assumes validated inputs.
  This scope measures each existing entry point rather than isolating compiler
  code generation. Strategy selection is outside the prepared timer. Partition
  speedups compare the same backend at the same P.
- **Features scope:** complete public feature extraction on raw input, with
  application caches disabled for both. Includes normalization, filter
  generation and packing. This is the comparable end-to-end transformation.
- Additional scratch payload is reported separately from process peak RSS.
  The latter includes imports/JIT/runtime allocations and is not presented as
  a per-strategy memory delta.

See [the full measured table](results/temporal/report.md),
[CSV](results/temporal/measurements.csv), and
[raw timings and metadata](results/temporal/measurements.json).


## Recorded local validation and results

Measured on Apple M1 Pro, macOS arm64, NumPy 2.4.6, Numba 0.67.0, sktime
1.1.0, with the OpenMP Numba backend. Nine warmed repetitions, randomized
order; table below uses four requested CPU threads and prepared inputs.
Times are milliseconds; speedup compares the original and combined paths at
the same thread count. All 264 measurements and quartiles are in the full report.

| Backend | Case | Original ms | 2 chunks ms | 4 chunks ms | 4-chunk speedup | Auto chunks / speedup |
|---|---|---:|---:|---:|---:|---:|
| cpp | single_long | 0.9895 | 0.6690 | 0.4866 | 2.03x | 4 / 2.03x |
| cpp | two_long_kernels | 1.0139 | 0.4499 | 0.4496 | 2.25x | 2 / 2.25x |
| cpp | saturated_short | 3.9931 | 4.4134 | 4.4912 | 0.89x | 1 / 1.00x |
| numba | single_long | 0.8801 | 0.5149 | 0.3074 | 2.86x | 4 / 2.81x |
| numba | two_long_kernels | 0.6617 | 0.3834 | 0.3839 | 1.72x | 2 / 1.73x |
| numba | saturated_short | 5.3051 | 5.4978 | 5.6988 | 0.93x | 1 / 1.00x |

Full public feature extraction for `single_long`, cpp: 2.6087 ms -> 2.2347 ms with four chunks (1.17x). Full public feature extraction for `single_long`, numba: 2.5109 ms -> 2.0536 ms with four chunks (1.22x). 

The long single-kernel case benefits from real temporal splitting. Two long
kernels generally need only two chunks at four workers; extra chunks add
scratch without useful improvement. Short convolutions regress when splitting
is forced: thread dispatch and reduction dominate. Already-saturated kernels
also regress modestly; auto keeps the original map. The refined sample-aware
policy chooses two chunks for underfilled C++ tiles in `many_samples` and
`mixed_long`, recovering about 1.56x and 1.74x respectively. Those batch gains
also include finer sample/kernel scheduling.

Validation: **179 repository tests passed**. The original Numba branch suite
also passed against the integrated module: **14 passed, 1 skipped** (optional
external C++ solver). Six standalone GunPoint runs, both engines with 1, 2
and auto chunks, all returned Accuracy 0.98 and F1 0.9795918367346939.
Their logs and `cli_checks.json` are stored beside the benchmark.

Within each backend, temporal PPV/MAX results equalled their original outputs
exactly in tests and all benchmark checks. Cross-engine PPV was identical on
the measured inputs; the largest existing cross-engine MAX rounding difference
was 3.814697265625e-6. This is not introduced by temporal splitting.

Recommendation: use `temporal_chunks="auto"` for mixed workloads on this
machine, preserving the default 1 for backward compatibility. Force 2/4 when
profiling a specific long, underfilled workload. Retune the conservative cost
thresholds and test additional thread counts on the destination HPC CPU; the
recorded results do not establish a universal optimal split.
