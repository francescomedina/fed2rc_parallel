"""Reproducible comparison of the original Fed2RC hot paths and native maps.

NUMBA_NUM_THREADS=8 NUMBA_THREADING_LAYER=omp python benchmark_comparison.py
Requires the project numerical dependencies and matplotlib. No UCR/Fluke run is
implied: these are synthetic-data measurements of the two changed hot paths.
The original functions are extracted from the recorded Git revision, not
reimplemented for this comparison. BLAS uses one thread in every variant.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

# Configure native libraries before matplotlib/NumPy/SciPy can initialize them.
# In particular, threadpoolctl does not report/control Apple's Accelerate;
# an empty BLAS inventory is NOT proof that it uses a single thread.
BLAS_ENV = {key: "1" for key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "MKL_NUM_THREADS", "BLIS_NUM_THREADS")}
os.environ.update(BLAS_ENV)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
import sktime
from sklearn.metrics import accuracy_score
from sktime.transformations.panel.rocket import Rocket
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from parallel_utils import FeatureCache, select_lambda, transform_features

BASELINE = "209009c55b6b4d84b748ac5043ab48ee568fe4e5"


def original_functions(revision):
    sources = {name: subprocess.check_output(
        ["git", "show", f"{revision}:code/{name}.py"], cwd=ROOT, text=True)
        for name in ("utils", "fed2rc")}
    utility = ast.parse(sources["utils"])
    nodes = [node for node in utility.body
             if getattr(node, "name", "") in ("RocketKernel", "transform_seeds")]
    scope = {"np": np, "Rocket": Rocket}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "original_utils", "exec"), scope)

    fed = ast.parse(sources["fed2rc"])
    client = next(node for node in fed.body if getattr(node, "name", "") == "ClientFed2RC")
    tune = next(node for node in client.body if getattr(node, "name", "") == "tune_lambda")
    loop = next(node for node in tune.body if isinstance(node, ast.For))
    # Only the grid is made an argument, to permit weak scaling. Its loop body,
    # normalization, accuracy_score call and sequential tie reduction are exact.
    loop.iter = ast.Name(id="lambdas", ctx=ast.Load())
    wrapper = ast.parse(
        "def original_solver(A, b, X, y, lambdas):\n"
        "    A_global = A_minus_client = A\n"
        "    b_minus_client = b\n"
        "    X_trans = X\n"
        "    best_accuracy = tmp_lambda = 0\n"
        "    return tmp_lambda, best_accuracy\n"
    )
    wrapper.body[0].body.insert(-1, loop)
    ast.fix_missing_locations(wrapper)
    exec(compile(wrapper, "original_lambda_loop", "exec"),
         solver_scope := {"np": np, "accuracy_score": accuracy_score})

    # The Fluke variant already had a cache. Measure that warm-cache path too,
    # rather than attributing all reuse benefits to the new implementation.
    cache_nodes = [node for node in fed.body if getattr(node, "name", "") in
                   ("get_from_cache", "put_in_cache", "transform", "transform_seeds")]
    future = ast.parse("from __future__ import annotations").body
    cache_scope = {"np": np, "Rocket": Rocket, "FEAT_CACHE": {}}
    exec(compile(ast.Module(body=future + cache_nodes, type_ignores=[]),
                 "original_fluke_cache", "exec"), cache_scope)
    return scope["transform_seeds"], solver_scope["original_solver"], cache_scope, sources


def collect(cases, repeats, rng):
    """Warm each exact input/configuration, randomize order, verify outside timing."""
    samples = {name: [] for name in cases}
    for threads, prepare, call, check in cases.values():
        numba.set_num_threads(threads)
        prepare()
        check(call())
    for _ in range(repeats):
        names = list(cases)
        rng.shuffle(names)
        for name in names:
            threads, prepare, call, check = cases[name]
            numba.set_num_threads(threads)
            prepare()
            start = perf_counter()
            result = call()
            elapsed = perf_counter() - start
            check(result)
            samples[name].append(elapsed)
    return {name: {"seconds": values, "median": float(np.median(values)),
                   "q25": float(np.quantile(values, .25)),
                   "q75": float(np.quantile(values, .75))}
            for name, values in samples.items()}


def plot(data, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.subplots_adjust(top=.81, bottom=.17, hspace=.58, wspace=.29)
    fig.suptitle("Fed2RC | originale e versione parallela", fontsize=21, fontweight="bold", y=.975)
    fig.text(.5, .913, "Hot path su dati sintetici · cache disabilitata nei confronti principali\n"
             f"{data['metadata']['machine']} · BLAS richiesto: 1 thread · mediana di {data['metadata']['repeats']} prove · JIT escluso",
             ha="center", fontsize=11, color="#475569")
    strong = data["strong"]
    pbar = 4 if 4 in data["threads"] else data["threads"][-1]
    for ax, key, title in ((axes[0, 0], "rocket", "Trasformazione ROCKET"),
                           (axes[0, 1], "lambda", "Ricerca di 100 lambda")):
        records = [strong[f"{key}_original"], strong[f"{key}_1"], strong[f"{key}_{pbar}"]]
        ms = np.array([v["median"] * 1000 for v in records])
        errors = np.array([[max(0, v["median"] - v["q25"]) * 1000 for v in records],
                           [max(0, v["q75"] - v["median"]) * 1000 for v in records]])
        bars = ax.bar(["Originale\n1 thread", "Nuovo\n1 thread", f"Nuovo\n{pbar} thread"],
                      ms, yerr=errors, capsize=5, color=["#94a3b8", "#38bdf8", "#0f766e"], width=.6)
        for bar, value in zip(bars, ms):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ms) * .06,
                    f"{value:.1f} ms", ha="center", fontweight="bold")
        ax.set_ylim(0, max(ms) * 1.35)
        ax.set_ylabel("Tempo [ms] — meno è meglio")
        ax.set_title(title + f"\nVantaggio sul codice originale: {ms[0] / ms[2]:.2f}×", pad=14)
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)

    ax = axes[1, 0]
    ps = data["threads"]
    for key, label, color in (("rocket", "ROCKET", "#2563eb"), ("lambda", "Lambda", "#0f766e")):
        one = strong[f"{key}_1"]["median"]
        ax.plot(ps, [one / strong[f"{key}_{p}"]["median"] for p in ps], "o-",
                label=label, color=color, lw=2.4, ms=7)
    ax.plot(ps, ps, "--", color="#94a3b8", label="Ideale: S(p) = p")
    ax.set_title("Strong scaling | problema fisso", pad=13)
    ax.set_ylabel("Speedup del nuovo codice: T(1) / T(p)")
    ax.set_xlabel("Thread nativi p")
    ax.set_xticks(ps)
    ax.grid(alpha=.16)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    weak = data["weak_lambda"]
    base = weak["1"]["median"]
    ax.plot(ps, [weak[str(p)]["median"] / base for p in ps], "o-", color="#0f766e", lw=2.4, ms=7)
    ax.axhline(1, color="#94a3b8", ls="--", label="Ideale: tempo costante")
    ax.set_title("Weak scaling | 100 lambda per thread", pad=13)
    ax.set_ylabel("Tempo normalizzato: T(p, 100p) / T(1, 100)")
    ax.set_xlabel("Thread p / lambda totali")
    ax.set_xticks(ps, [f"{p}\n{100*p}" for p in ps])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.16)
    ax.legend(frameon=False)
    fig.text(.07, .035,
             "ROCKET: 512 serie × 256 campioni, 128 seed. Lambda: sistemi 128×128, 4 classi, 128 serie di validazione.\n"
             "Barre d’errore: quartili 25–75%. Nuovo/1 thread separa ottimizzazione seriale e parallelismo.\n"
             "Sono misure dei due hot path: non rappresentano il tempo completo di training federato/UCR.",
             fontsize=10, color="#475569")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"comparison.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "hpc_comparison")
    args = parser.parse_args()
    if args.repeats < 3 or args.threads[0] != 1 or any(
            p < 1 or p > numba.config.NUMBA_NUM_THREADS for p in args.threads):
        parser.error("Need >=3 repetitions and thread counts starting at 1 within NUMBA_NUM_THREADS")
    args.output.mkdir(parents=True, exist_ok=True)
    original_transform, original_solver, original_cache, sources = original_functions(args.baseline)
    rng = np.random.default_rng(20260908)
    X = rng.normal(size=(512, 256))
    seeds = np.arange(128)
    expected_features = original_transform(X, seeds, X.shape[1])
    # A realistic ridge system built from the same transformed features. Setup
    # and data generation are excluded identically from all solver measurements.
    features = expected_features.astype(np.float64)
    train, validation = features[:384], np.ascontiguousarray(features[384:])
    labels = rng.integers(0, 4, len(features))
    A = train.T @ train
    b = train.T @ np.eye(4)[labels[:384]]
    y = labels[384:]
    lambdas = np.logspace(-3, 0, 100)
    expected_best = original_solver(A, b, validation, y, lambdas)

    def check_features(value):
        np.testing.assert_array_equal(value, expected_features)

    def check_best(value):
        assert tuple(value[:2]) == expected_best

    noop = lambda: None
    cases = {
        "rocket_original": (1, noop, lambda: original_transform(X, seeds, X.shape[1]), check_features),
        "lambda_original": (1, noop, lambda: original_solver(A, b, validation, y, lambdas), check_best),
    }
    for p in args.threads:
        cases[f"rocket_{p}"] = (p, noop, lambda: transform_features(X, seeds, caching=False), check_features)
        cases[f"lambda_{p}"] = (p, noop, lambda: select_lambda(A, b, validation, y, lambdas), check_best)
    print("Measuring original/new hot paths and strong scaling...", flush=True)
    strong = collect(cases, args.repeats, rng)

    cache = FeatureCache()
    cache_threads = 4 if 4 in args.threads else args.threads[-1]
    def prefill():
        transform_features(X, seeds, cache=cache)
    def old_prefill():
        original_cache["transform_seeds"](X, 0, seeds)
    cache_cases = {
        "new_cold": (cache_threads, cache.clear, lambda: transform_features(X, seeds, cache=cache), check_features),
        "new_warm": (cache_threads, prefill, lambda: transform_features(X, seeds, cache=cache), check_features),
        "original_fluke_warm": (1, old_prefill,
                                lambda: original_cache["transform_seeds"](X, 0, seeds), check_features),
    }
    print("Measuring cold/warm caches, including the original Fluke cache...", flush=True)
    caches = collect(cache_cases, args.repeats, rng)

    weak_cases = {}
    for p in args.threads:
        grid = np.logspace(-3, 0, 100 * p)
        expected = original_solver(A, b, validation, y, grid)
        def check(value, expected=expected):
            assert tuple(value[:2]) == expected
        weak_cases[str(p)] = (p, noop, lambda grid=grid: select_lambda(A, b, validation, y, grid), check)
    print("Measuring weak scaling with 100 lambda tasks per thread...", flush=True)
    weak = collect(weak_cases, args.repeats, rng)
    pools = threadpool_info()
    assert all(pool["num_threads"] == 1 for pool in pools if pool["user_api"] == "blas")
    machine = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        machine = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    metadata = dict(date_utc=datetime.now(timezone.utc).isoformat(), baseline=args.baseline,
                    baseline_source_sha256={name: hashlib.sha256(src.encode()).hexdigest()
                                            for name, src in sources.items()},
                    machine=machine, platform=platform.platform(), python=sys.version,
                    numpy=np.__version__, scipy=scipy.__version__, sktime=sktime.__version__,
                    numba=numba.__version__, threading_layer=numba.threading_layer(),
                    threadpools=pools, repeats=args.repeats, seed=20260908,
                    blas_environment=BLAS_ENV,
                    numpy_blas=np.__config__.CONFIG["Build Dependencies"]["blas"],
                    scipy_blas=scipy.__config__.CONFIG["Build Dependencies"]["blas"],
                    blas_note="Accelerate thread count is requested via environment; threadpoolctl cannot verify it",
                    timing="Wall clock, warm JIT, randomized order, checks/setup outside timing",
                    scope="Synthetic hot paths, not full federated training; BLAS fixed at 1",
                    workload=dict(samples=512, timepoints=256, seeds=128, classes=4,
                                  train_samples=384, validation_samples=128, lambdas=100),
                    cache_threads=cache_threads)
    data = dict(metadata=metadata, threads=args.threads, strong=strong, cache=caches, weak_lambda=weak)
    (args.output / "measurements.json").write_text(json.dumps(data, indent=2) + "\n")
    # Raw timings permit alternative statistics without rerunning measurements.
    rows = ["section,case,repetition,seconds"]
    for section, records in (("strong", strong), ("cache", caches), ("weak_lambda", weak)):
        for name, record in records.items():
            rows.extend(f"{section},{name},{i},{t:.12g}" for i, t in enumerate(record["seconds"]))
    (args.output / "measurements.csv").write_text("\n".join(rows) + "\n")
    plot(data, args.output)
    print(json.dumps({"strong_ms": {k: round(v["median"] * 1000, 3) for k, v in strong.items()},
                      "cache_ms": {k: round(v["median"] * 1000, 3) for k, v in caches.items()},
                      "weak_ms": {k: round(v["median"] * 1000, 3) for k, v in weak.items()},
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
