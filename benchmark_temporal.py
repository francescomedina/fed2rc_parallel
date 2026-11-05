"""Reproducible kernel-only versus kernel+temporal ROCKET measurements.

NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python benchmark_temporal.py

Prepared scope includes output/scratch allocation and API/packing overhead, but
excludes normalization/kernel generation. Features scope includes the public
helpers, with application caches disabled for both engines. No JIT is timed.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
from time import perf_counter

# Set before importing any numerical runtime. Explicit external limits are kept.
os.environ.setdefault("NUMBA_NUM_THREADS", str(min(8, os.cpu_count() or 1)))
os.environ.setdefault("OMP_NUM_THREADS", os.environ["NUMBA_NUM_THREADS"])
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
os.environ["OMP_DYNAMIC"] = "FALSE"
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numba
import numpy as np
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
import openmp_utils as cpp
import parallel_utils as nb
from temporal import (MIN_OUTPUTS_PER_CHUNK, MIN_TAP_OPERATIONS_PER_WORKER,
                      MIN_BATCH_OUTPUTS_PER_CHUNK, resolve_temporal_chunks)


# Few long convolutions expose the new parallelism; saturated/short workloads
# expose overhead and must remain on the original path under auto.
CASES = {
    "single_short": (1, 128, 1),
    "single_medium": (1, 8193, 1),
    "single_long": (1, 262145, 1),
    "two_long_kernels": (1, 131073, 2),
    "eight_long_kernels": (1, 65537, 8),
    "many_kernels": (1, 4097, 64),
    "many_samples": (64, 1025, 8),
    "saturated_short": (128, 257, 128),
    "mixed_long": (8, 16385, 16),
}


def kernel_arrays(timepoints, kernels):
    weights = np.zeros((kernels, 11), np.float32)
    lengths, dilation, padding = (np.empty(kernels, np.int32) for _ in range(3))
    biases = np.empty(kernels, np.float32)
    for seed in range(kernels):
        w, length, bias, dil, pad = cpp._kernel(timepoints, seed)
        weights[seed, :length] = w
        lengths[seed], biases[seed], dilation[seed], padding[seed] = length, bias, dil, pad
    return weights, lengths, biases, dilation, padding


def choose(backend, chunks, X, lengths, dilation, padding, threads):
    jobs = len(lengths) if backend == "numba" else (
        ((len(X) + 31) // 32) * ((len(lengths) + 7) // 8)
    )
    return resolve_temporal_chunks(chunks, X.shape[1], lengths, dilation, padding,
                                   threads, jobs, samples=len(X))


def prepared_call(backend, data, requested, threads):
    X, weights, lengths, biases, dilation, padding = data
    chunks = choose(backend, requested, X, lengths, dilation, padding, threads)
    if backend == "cpp":
        call = lambda: cpp.native.transform(X, weights, biases, lengths, dilation,
                                            padding, False, threads, chunks)
    else:
        def call():
            # Same original Numba storage/feature order; scratch is call-local.
            out = np.empty((len(lengths), len(X), 2), np.float32)
            if chunks == 1:
                nb._transform_map(X, weights, lengths, biases, dilation, padding, out)
            else:
                counts = np.empty((len(lengths), len(X), chunks), np.int64)
                maxima = np.empty(counts.shape, np.float32)
                nb._transform_temporal(X, weights, lengths, biases, dilation, padding,
                                       out, counts, maxima)
            return np.concatenate(list(out), axis=1)
    # Actual partial-array payload: int64 counts + float32 maxima. No per-output
    # response vectors. C++ uses equally sized buffers; Python metadata excluded.
    scratch = 0 if chunks == 1 else len(X) * len(lengths) * chunks * 12
    return call, chunks, scratch


def measure_group(configurations, repeats, target_seconds, rng):
    """Warm all specializations, verify exact equivalence, then randomize order."""
    expected = {}
    batches = {}
    warmup_start = perf_counter()
    for name, (backend, threads, call, chunks, scratch) in configurations.items():
        numba.set_num_threads(threads)
        actual = call()
        if backend not in expected:
            expected[backend] = actual
        np.testing.assert_array_equal(actual, expected[backend])
        start = perf_counter()
        call()
        pilot = perf_counter() - start
        batches[name] = max(1, min(512, int(np.ceil(target_seconds / max(pilot, 1e-9)))))
    warmup_seconds = perf_counter() - warmup_start
    observations = {name: [] for name in configurations}
    orders = []
    for _ in range(repeats):
        order = list(configurations)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            backend, threads, call, chunks, scratch = configurations[name]
            numba.set_num_threads(threads)
            start = perf_counter()
            for _ in range(batches[name]):
                result = call()
            elapsed = (perf_counter() - start) / batches[name]
            observations[name].append(elapsed)
            # Checks are outside timing; catches incomplete/non-deterministic writes.
            np.testing.assert_array_equal(result, expected[backend])
    cross = {
        "ppv_exact": bool(np.array_equal(expected["cpp"][:, ::2], expected["numba"][:, ::2])),
        "maximum_max_abs_difference": float(np.max(np.abs(
            expected["cpp"][:, 1::2] - expected["numba"][:, 1::2]
        ))),
    }
    return observations, batches, orders, warmup_seconds, cross


def worker_probe(backend, threads):
    """Record owners from the actual single-series/single-kernel temporal path."""
    X = np.random.default_rng(77).normal(size=(1, 65537)).astype(np.float32)
    weights, lengths, biases, dilation, padding = kernel_arrays(X.shape[1], 1)
    owners = np.full(threads, -1, np.int32)
    numba.set_num_threads(threads)
    if backend == "cpp":
        cpp.native.transform(X, weights, biases, lengths, dilation, padding,
                             False, threads, threads, owners)
    else:
        out = np.empty((1, 1, 2), np.float32)
        counts = np.empty((1, 1, threads), np.int64)
        nb._transform_temporal(X, weights, lengths, biases, dilation, padding,
                               out, counts, np.empty(counts.shape, np.float32), owners)
    observed = sorted(int(i) for i in np.unique(owners))
    if observed != list(range(threads)):
        raise AssertionError(f"{backend}: requested {threads} workers, observed {observed}")
    return observed


def write_report(output, report):
    output.mkdir(parents=True, exist_ok=True)
    (output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = report["measurements"]
    fields = [key for key in rows[0] if key != "seconds"]
    with (output / "measurements.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    lines = ["# ROCKET temporal partition benchmark", "",
             f"Machine: {report['metadata']['cpu']}; {report['metadata']['platform']}.",
             f"Median of {report['metadata']['repeats']} warmed repetitions; compilation excluded.",
             "Speedup is kernel-only time / combined time at the **same CPU thread count**.",
             "Scratch is the additional allocated partial-array payload, not RSS.", ""]
    for scope in ["prepared", "features"]:
        selected = [r for r in rows if r["scope"] == scope]
        if not selected:
            continue
        lines += [f"## {scope}", "",
                  "| Backend | Case (N,T,K) | Threads | Original ms | 2 chunks ms (speedup) | 4 chunks ms (speedup) | Auto ms (chunks; speedup) | Scratch at 4 chunks KiB |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        groups = {}
        for row in selected:
            key = row["backend"], row["case"], row["threads"]
            groups.setdefault(key, {})[str(row["requested_chunks"])] = row
        for (backend, case, threads), group in groups.items():
            baseline = group["1"]
            def cell(key):
                row = group.get(key)
                return "—" if row is None else f"{row['median_ms']:.4f} ({row['speedup']:.2f}×)"
            auto = group.get("auto")
            auto_cell = "—" if auto is None else f"{auto['median_ms']:.4f} ({auto['effective_chunks']}; {auto['speedup']:.2f}×)"
            shape = f"{baseline['samples']},{baseline['timepoints']},{baseline['kernels']}"
            scratch = group.get("4", baseline)["scratch_bytes"] / 1024
            lines.append(f"| {backend} | {case} ({shape}) | {threads} | {baseline['median_ms']:.4f} | {cell('2')} | {cell('4')} | {auto_cell} | {scratch:.3f} |")
        lines.append("")
    lines += ["## Interpretation", "",
              "- `prepared`: identical normalized float32 input and identical pre-generated kernels. Includes allocations and each engine's existing boundary/packing costs. C++ validates native inputs; the internal Numba map assumes validated inputs.",
              "- Prepared strategy selection is outside timing; `features` includes the public auto-policy decision on every call.",
              "- `features`: full public transformation, identical raw data/seeds, application caches disabled. Includes normalization, sktime parameter generation, and feature assembly.",
              "- Within each engine, every strategy must equal its untouched kernel-only output **exactly**, for PPV and MAX, before and after timing.",
              "- Cross-engine PPV agreement/MAX rounding differences are recorded separately in JSON; existing fast-math differences are not attributed to temporal splitting.",
              "- Auto uses the existing kernel/tile job count and minimum output length. It returns 1 when workers are already occupied or chunks would be too short.",
              "- Worker-ID evidence comes from the actual temporal map with one sample and one kernel, not a separate empty parallel region.",
              "- RSS includes imports, JIT and runtime pools; only partial-array payload is attributed exactly to temporal splitting.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--target-ms", type=float, default=15)
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "temporal")
    args = parser.parse_args()
    if args.repeats < 3 or args.target_ms <= 0 or any(
        p < 1 or p > numba.config.NUMBA_NUM_THREADS for p in args.threads
    ):
        parser.error("need >=3 repeats, positive target time, and threads within NUMBA_NUM_THREADS")
    threads = sorted(set(args.threads))
    records, experiments = [], []
    order_rng = random.Random(20261105)
    start = perf_counter()
    for name in args.cases:
        case_index = list(CASES).index(name)
        n, t, k = CASES[name]
        X = np.random.default_rng(300 + case_index).normal(size=(n, t))
        normalized = np.require((X-X.mean(axis=1, keepdims=True)) / (
            X.std(axis=1, keepdims=True)+1e-8), dtype=np.float32, requirements=["C", "A"])
        weights, lengths, biases, dilation, padding = kernel_arrays(t, k)
        data = normalized, weights, lengths, biases, dilation, padding
        scopes = ["prepared"]
        if not args.skip_features and name in ["single_long", "saturated_short"]:
            scopes.append("features")
        for scope in scopes:
            for p in threads:
                configurations = {}
                for backend in ["cpp", "numba"]:
                    for requested in [1, 2, 4, "auto"]:
                        call, effective, scratch = prepared_call(backend, data, requested, p)
                        if scope == "features":
                            transform = cpp.transform_features if backend == "cpp" else nb.transform_features
                            keywords = dict(ppv_only=False, caching=False, temporal_chunks=requested)
                            if backend == "cpp":
                                keywords["threads"] = p
                            call = lambda transform=transform, keywords=keywords: transform(X, range(k), **keywords)
                        configurations[backend, str(requested)] = backend, p, call, effective, scratch
                observations, batches, orders, warmup, cross = measure_group(
                    configurations, args.repeats, args.target_ms / 1000, order_rng)
                experiment = dict(case=name, scope=scope, threads=p, cross_engine=cross,
                                  warmup_seconds=warmup,
                                  execution_orders=[[list(key) for key in order] for order in orders])
                experiments.append(experiment)
                for key, values in observations.items():
                    backend, strategy = key
                    effective, scratch = configurations[key][3:]
                    baseline = float(np.median(observations[backend, "1"]))
                    median = float(np.median(values))
                    records.append(dict(case=name, scope=scope, backend=backend, samples=n,
                                        timepoints=t, kernels=k, threads=p,
                                        requested_chunks=int(strategy) if strategy != "auto" else strategy,
                                        effective_chunks=effective, median_ms=median*1000,
                                        q25_ms=float(np.quantile(values, .25))*1000,
                                        q75_ms=float(np.quantile(values, .75))*1000,
                                        speedup=baseline/median, scratch_bytes=scratch,
                                        output_bytes=n*k*2*4, calls_per_repeat=batches[key],
                                        seconds=values))
                summary = [f"{r['backend']} c{r['requested_chunks']}={r['speedup']:.2f}x"
                           for r in records[-8:] if r['requested_chunks'] != 1]
                print(f"{name} {scope} threads={p}: " + ", ".join(summary), flush=True)
    probes = {backend: {str(p): worker_probe(backend, p) for p in threads if p > 1}
              for backend in ["cpp", "numba"]}
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak_rss *= 1024
    sources = ["code/parallel_utils.py", "code/temporal.py", "code/openmp_utils.py",
               "native/rocket.hpp", "native/bindings.cpp", "benchmark_temporal.py"]
    report = dict(metadata=dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(), cpu=cpu, platform=platform.platform(),
        logical_cpus=os.cpu_count(), python=platform.python_version(),
        versions={p: importlib.metadata.version(p) for p in ["numpy", "numba", "sktime", "scipy", "threadpoolctl"]},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        numba_threading_layer=numba.threading_layer(), native_runtime=cpp.native.runtime_info(),
        threadpools=threadpool_info(), repeats=args.repeats, target_ms=args.target_ms,
        auto_min_outputs_per_chunk=MIN_OUTPUTS_PER_CHUNK,
        auto_min_tap_operations_per_worker=MIN_TAP_OPERATIONS_PER_WORKER, worker_ids=probes,
        auto_batch_min_outputs_per_chunk=MIN_BATCH_OUTPUTS_PER_CHUNK,
        peak_process_rss_bytes=peak_rss, total_benchmark_seconds=perf_counter()-start,
        environment={name:os.environ.get(name) for name in ["OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMBA_THREADING_LAYER", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]},
        command=sys.argv, numba_baseline_source="numba-comparison:code/parallel_utils.py at 1f3ddba"),
        measurements=records, experiments=experiments)
    write_report(args.output, report)
    print(f"Report: {args.output/'report.md'}", flush=True)


if __name__ == "__main__":
    main()
