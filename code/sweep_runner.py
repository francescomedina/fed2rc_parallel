"""Sweep clients x threads x kernels for fed2rc_no_fluke.py, on the full UCR
archive each time, and check that parallelism buys speed without changing
the result.

For every (clients, threads, kernels) combination this:
  1. Runs `python fed2rc_no_fluke.py --clients C --kernels K --debug` with
     NUMBA_NUM_THREADS=T, timing the whole run.
  2. Compares the run's averaged per-dataset Accuracy/F1 against the
     non-parallel baseline in ../results/Fed2RC/<split>/<K>kernels_results.csv
     (see analyze_results.py) to get a correctness pass/fail.
  3. Appends the outcome to results/parallel_sweep/measurements.csv and
     .json, so a long sweep can be interrupted and resumed.

At the end it plots seconds vs. threads (one line per clients/kernels
combination) and prints two verdicts:
  - ACCURACY MATCHES BASELINE (all configs within --tol of the non-parallel
    numbers)
  - SPEEDUP WITH THREADS (time does not increase as threads increase, for
    every clients/kernels group)

Each full-archive run takes hours on this project's UCR sweep; the default
grid (2 client counts x 3 thread counts x 1 kernel count = 6 runs) is already
a multi-run commitment. Use --dry-run to see the planned grid and estimated
command lines before committing to it.

Usage:
    python sweep_runner.py --dry-run
    python sweep_runner.py --clients-list 2,4 --threads-list 1,2,4 --kernels-list 100
"""

import argparse
import itertools
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from analyze_results import compare_run

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
DEFAULT_OUT_DIR = ROOT / "results" / "parallel_sweep"

# Pin native BLAS backends to one thread each so only the Numba maps this
# project parallelizes are varied by NUMBA_NUM_THREADS -- otherwise BLAS's
# own threading would confound the speed comparison. Mirrors
# benchmark_comparison.py's convention.
BLAS_ENV = {k: "1" for k in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
                              "MKL_NUM_THREADS", "BLIS_NUM_THREADS")}


def _int_list(text: str) -> list[int]:
    return [int(v) for v in text.split(",") if v.strip()]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_state(json_path: Path) -> dict:
    if json_path.exists():
        return json.loads(json_path.read_text())
    return {"metadata": None, "runs": []}


def _save_state(json_path: Path, state: dict) -> None:
    json_path.write_text(json.dumps(state, indent=2, default=str))


def _csv_header_if_needed(csv_path: Path) -> None:
    if not csv_path.exists():
        csv_path.write_text(
            "timestamp_utc,clients,threads,kernels,seconds,returncode,"
            "all_match,n_compared,n_matched,n_mismatched,"
            "mean_acc_diff,mean_f1_diff,max_acc_diff,log_path\n"
        )


def _append_csv_row(csv_path: Path, row: dict) -> None:
    fields = ["timestamp_utc", "clients", "threads", "kernels", "seconds", "returncode",
              "all_match", "n_compared", "n_matched", "n_mismatched",
              "mean_acc_diff", "mean_f1_diff", "max_acc_diff", "log_path"]
    with open(csv_path, "a") as f:
        f.write(",".join(str(row[k]) for k in fields) + "\n")


def run_one(clients: int, threads: int, kernels: int, rounds: int,
            split: str, tol: float, out_dir: Path, python_exe: str) -> dict:
    log_path = out_dir / "logs" / f"c{clients}_t{threads}_k{kernels}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(BLAS_ENV)
    env["NUMBA_NUM_THREADS"] = str(threads)
    env["NUMBA_THREADING_LAYER"] = "omp"

    cmd = [python_exe, "fed2rc_no_fluke.py",
           "--clients", str(clients), "--kernels", str(kernels),
           "--rounds", str(rounds), "--debug"]

    print(f"\n>>> clients={clients} threads={threads} kernels={kernels} "
          f"-> {log_path.relative_to(ROOT)}")
    print("    " + " ".join(f"{k}={v}" for k, v in
                             {"NUMBA_NUM_THREADS": threads, "NUMBA_THREADING_LAYER": "omp"}.items())
          + " " + " ".join(cmd))

    start = perf_counter()
    with open(log_path, "w") as log_file:
        proc = subprocess.run(cmd, cwd=CODE_DIR, env=env,
                               stdout=log_file, stderr=subprocess.STDOUT)
    seconds = perf_counter() - start
    print(f"    finished in {seconds:.1f}s (returncode={proc.returncode})")

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "clients": clients, "threads": threads, "kernels": kernels,
        "seconds": seconds, "returncode": proc.returncode,
        "all_match": False, "n_compared": 0, "n_matched": 0, "n_mismatched": 0,
        "mean_acc_diff": float("nan"), "mean_f1_diff": float("nan"),
        "max_acc_diff": float("nan"),
        "log_path": str(log_path.relative_to(ROOT)),
    }

    if proc.returncode != 0:
        print(f"    !! run failed (see {log_path}); skipping correctness check")
        return result

    try:
        cmp = compare_run(log_path, kernels, split, tol)
    except Exception as exc:
        print(f"    !! comparison failed: {exc}")
        return result

    result.update({
        "all_match": cmp["all_match"], "n_compared": cmp["n_compared"],
        "n_matched": cmp["n_matched"], "n_mismatched": cmp["n_mismatched"],
        "mean_acc_diff": cmp["mean_acc_diff"], "mean_f1_diff": cmp["mean_f1_diff"],
        "max_acc_diff": cmp["max_acc_diff"],
    })
    print(f"    accuracy match vs {split} baseline: {cmp['all_match']} "
          f"({cmp['n_matched']}/{cmp['n_compared']} datasets, "
          f"mean|acc diff|={cmp['mean_acc_diff']:.4f})")
    return result


def plot_results(runs: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in runs if r["returncode"] == 0]
    if not ok:
        print("No successful runs to plot.")
        return

    groups: dict[tuple, list[dict]] = {}
    for r in ok:
        groups.setdefault((r["clients"], r["kernels"]), []).append(r)

    fig, ax = plt.subplots(figsize=(7, 5))
    for (clients, kernels), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["threads"])
        threads = [r["threads"] for r in rows]
        seconds = [r["seconds"] for r in rows]
        marker = "o" if all(r["all_match"] for r in rows) else "x"
        ax.plot(threads, seconds, marker=marker,
                label=f"clients={clients}, kernels={kernels}")
    ax.set_xlabel("NUMBA_NUM_THREADS")
    ax.set_ylabel("Wall-clock seconds (full UCR archive)")
    ax.set_title("Fed2RC parallel sweep: threads vs. runtime\n"
                  "(o = accuracy matched baseline, x = mismatch)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_results.png", dpi=150)
    fig.savefig(out_dir / "sweep_results.svg")
    plt.close(fig)
    print(f"Wrote {out_dir / 'sweep_results.png'} and .svg")


def summarize(runs: list[dict]) -> bool:
    ok = [r for r in runs if r["returncode"] == 0]
    failed = [r for r in runs if r["returncode"] != 0]
    accuracy_ok = bool(ok) and all(r["all_match"] for r in ok)

    groups: dict[tuple, list[dict]] = {}
    for r in ok:
        groups.setdefault((r["clients"], r["kernels"]), []).append(r)
    speedup_ok = True
    for (clients, kernels), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["threads"])
        if len(rows) < 2:
            continue
        fastest_more_threads = rows[-1]["seconds"] <= rows[0]["seconds"]
        speedup_ok = speedup_ok and fastest_more_threads
        print(f"  clients={clients} kernels={kernels}: "
              f"{rows[0]['threads']}thr={rows[0]['seconds']:.1f}s -> "
              f"{rows[-1]['threads']}thr={rows[-1]['seconds']:.1f}s "
              f"({'faster or equal' if fastest_more_threads else 'SLOWER'})")

    print(f"\nCompleted: {len(ok)}/{len(runs)}  Failed: {len(failed)}")
    print(f"ACCURACY MATCHES BASELINE: {accuracy_ok}")
    print(f"SPEEDUP WITH THREADS: {speedup_ok}")
    print(f"PARALLEL SPEEDUP WITH MATCHING ACCURACY: {accuracy_ok and speedup_ok}")
    return accuracy_ok and speedup_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clients-list", type=_int_list, default=[2, 4])
    parser.add_argument("--threads-list", type=_int_list, default=[1, 2, 4])
    parser.add_argument("--kernels-list", type=_int_list, default=[100],
                        help="Must have a matching baseline CSV (100, 1000 or 10000)")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--split", choices=["IID", "Non-IID"], default="IID")
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true",
                        help="Re-run configs even if already recorded in the JSON state")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned grid and commands, run nothing")
    args = parser.parse_args()

    grid = list(itertools.product(args.clients_list, args.threads_list, args.kernels_list))
    print(f"Planned grid: {len(grid)} full-UCR-archive runs")
    for clients, threads, kernels in grid:
        print(f"  clients={clients} threads={threads} kernels={kernels} rounds={args.rounds}")
    if args.dry_run:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "measurements.csv"
    json_path = args.out_dir / "measurements.json"
    _csv_header_if_needed(csv_path)
    state = _load_state(json_path)
    state["metadata"] = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "machine": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "args": vars(args) | {"clients_list": args.clients_list,
                               "threads_list": args.threads_list,
                               "kernels_list": args.kernels_list,
                               "out_dir": str(args.out_dir), "python": args.python},
    }

    done = {(r["clients"], r["threads"], r["kernels"]) for r in state["runs"]
            if r["returncode"] == 0} if not args.force else set()

    for clients, threads, kernels in grid:
        if (clients, threads, kernels) in done:
            print(f"\n>>> clients={clients} threads={threads} kernels={kernels}: "
                  f"already recorded, skipping (use --force to re-run)")
            continue
        result = run_one(clients, threads, kernels, args.rounds, args.split,
                          args.tol, args.out_dir, args.python)
        state["runs"].append(result)
        _save_state(json_path, state)
        _append_csv_row(csv_path, result)

    plot_results(state["runs"], args.out_dir)
    summarize(state["runs"])


if __name__ == "__main__":
    main()
