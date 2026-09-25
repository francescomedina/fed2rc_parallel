"""Sweep clients x threads x kernels for fed2rc_no_fluke.py, on the full UCR
archive each time, and check that parallelism buys speed without changing
the result -- against a freshly-run classic (pre-parallelization, serial)
baseline instead of the static CSVs under results/Fed2RC.

For every (clients, kernels) pair this first runs the serial baseline once:
    python fed2rc_no_fluke.py --clients C --kernels K --engine classic --debug
(NUMBA_NUM_THREADS is irrelevant to --engine classic, so it is only run once
per clients/kernels combination, not once per thread count.)

Then, for every threads value in --threads-list, it runs:
    NUMBA_NUM_THREADS=T python fed2rc_no_fluke.py --clients C --kernels K --engine parallel --debug
times it, and compares its per-dataset Accuracy/F1/Time directly against
that clients/kernels pair's classic log (see analyze_results.compare_live)
-- same environment, same sklearn/sktime/numpy versions, same split, so this
is a live apples-to-apples check, and it yields a real speedup ratio
(classic seconds / parallel seconds) instead of only a pass/fail flag.

Every outcome (including the classic baseline runs, recorded with
threads=None) streams into results/parallel_sweep/measurements.csv and
.json, so a long sweep can be interrupted and resumed.

At the end it plots seconds vs. threads (one line per clients/kernels
combination, with the classic baseline as a dashed horizontal line) and
prints three verdicts:
  - ACCURACY MATCHES CLASSIC (every parallel config within --tol of its
    matched classic run)
  - SPEEDUP WITH THREADS (time does not increase as threads increase, for
    every clients/kernels group)
  - BEATS CLASSIC (the fastest thread count is actually faster than the
    serial classic baseline, for every clients/kernels group)

Each full-archive run takes hours; the default grid (2 client counts x
(1 classic + 3 parallel thread counts) x 1 kernel count = 8 runs) is already
a multi-run commitment. Use --dry-run to see the planned grid first.

Usage:
    python sweep_runner.py --dry-run
    python sweep_runner.py --clients-list 2,4 --threads-list 1,2,4 --kernels-list 100
"""

from __future__ import annotations

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

from analyze_results import compare_live

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
DEFAULT_OUT_DIR = ROOT / "results" / "parallel_sweep"

# Pin native BLAS backends to one thread each so only the Numba maps this
# project parallelizes are varied by NUMBA_NUM_THREADS -- otherwise BLAS's
# own threading would confound the speed comparison. Mirrors
# benchmark_comparison.py's convention.
BLAS_ENV = {k: "1" for k in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
                              "MKL_NUM_THREADS", "BLIS_NUM_THREADS")}

CSV_FIELDS = ["timestamp_utc", "clients", "engine", "threads", "kernels",
              "seconds", "returncode", "all_match", "n_compared", "n_matched",
              "n_mismatched", "mean_acc_diff", "mean_f1_diff", "overall_speedup",
              "log_path"]


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
        csv_path.write_text(",".join(CSV_FIELDS) + "\n")


def _append_csv_row(csv_path: Path, row: dict) -> None:
    with open(csv_path, "a") as f:
        f.write(",".join(str(row[k]) for k in CSV_FIELDS) + "\n")


def run_engine(engine: str, clients: int, kernels: int, rounds: int,
               threads: int | None, out_dir: Path, python_exe: str) -> tuple[Path, dict]:
    tag = f"c{clients}_k{kernels}_{engine}" + (f"_t{threads}" if threads else "")
    log_path = out_dir / "logs" / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(BLAS_ENV)
    if threads is not None:
        env["NUMBA_NUM_THREADS"] = str(threads)
        env["NUMBA_THREADING_LAYER"] = "omp"

    cmd = [python_exe, "fed2rc_no_fluke.py",
           "--clients", str(clients), "--kernels", str(kernels),
           "--rounds", str(rounds), "--engine", engine, "--debug"]

    label = f"clients={clients} kernels={kernels} engine={engine}" + \
            (f" threads={threads}" if threads else "")
    print(f"\n>>> {label} -> {log_path.relative_to(ROOT)}")

    start = perf_counter()
    with open(log_path, "w") as log_file:
        proc = subprocess.run(cmd, cwd=CODE_DIR, env=env,
                               stdout=log_file, stderr=subprocess.STDOUT)
    seconds = perf_counter() - start
    print(f"    finished in {seconds:.1f}s (returncode={proc.returncode})")

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "clients": clients, "engine": engine, "threads": threads if threads else "",
        "kernels": kernels, "seconds": seconds, "returncode": proc.returncode,
        "all_match": "", "n_compared": "", "n_matched": "", "n_mismatched": "",
        "mean_acc_diff": "", "mean_f1_diff": "", "overall_speedup": "",
        "log_path": str(log_path.relative_to(ROOT)),
    }
    return log_path, row


def run_group(clients: int, kernels: int, threads_list: list[int], rounds: int,
              tol: float, out_dir: Path, python_exe: str, state: dict,
              csv_path: Path, json_path: Path, force: bool) -> None:
    done = {(r["clients"], r["engine"], r["threads"], r["kernels"])
            for r in state["runs"] if r["returncode"] == 0} if not force else set()

    classic_key = (clients, "classic", "", kernels)
    if classic_key in done:
        classic_row = next(r for r in state["runs"]
                            if (r["clients"], r["engine"], r["threads"], r["kernels"]) == classic_key)
        classic_log = ROOT / classic_row["log_path"]
        print(f"\n>>> classic baseline clients={clients} kernels={kernels}: "
              f"already recorded, skipping (use --force to re-run)")
    else:
        classic_log, classic_row = run_engine("classic", clients, kernels, rounds,
                                              None, out_dir, python_exe)
        state["runs"].append(classic_row)
        _save_state(json_path, state)
        _append_csv_row(csv_path, classic_row)

    if classic_row["returncode"] != 0:
        print(f"    !! classic baseline failed; skipping parallel configs for "
              f"clients={clients} kernels={kernels}")
        return

    for threads in threads_list:
        key = (clients, "parallel", threads, kernels)
        if key in done:
            print(f"\n>>> clients={clients} kernels={kernels} threads={threads}: "
                  f"already recorded, skipping (use --force to re-run)")
            continue
        parallel_log, row = run_engine("parallel", clients, kernels, rounds,
                                       threads, out_dir, python_exe)
        if row["returncode"] == 0:
            try:
                cmp = compare_live(parallel_log, classic_log, tol)
                row.update({
                    "all_match": cmp["all_match"], "n_compared": cmp["n_compared"],
                    "n_matched": cmp["n_matched"], "n_mismatched": cmp["n_mismatched"],
                    "mean_acc_diff": cmp["mean_acc_diff"], "mean_f1_diff": cmp["mean_f1_diff"],
                    "overall_speedup": cmp.get("overall_speedup", ""),
                })
                print(f"    accuracy match vs classic: {cmp['all_match']} "
                      f"({cmp['n_matched']}/{cmp['n_compared']} datasets, "
                      f"mean|acc diff|={cmp['mean_acc_diff']:.4f}), "
                      f"speedup={cmp.get('overall_speedup', float('nan')):.2f}x, "
                      f"time={row['seconds']:.1f}s (classic={classic_row['seconds']:.1f}s)")
            except Exception as exc:
                print(f"    !! comparison failed: {exc}")
        state["runs"].append(row)
        _save_state(json_path, state)
        _append_csv_row(csv_path, row)


def plot_results(runs: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in runs if r["returncode"] == 0]
    parallel_ok = [r for r in ok if r["engine"] == "parallel"]
    classic_ok = {(r["clients"], r["kernels"]): r["seconds"]
                  for r in ok if r["engine"] == "classic"}
    if not parallel_ok:
        print("No successful parallel runs to plot.")
        return

    groups: dict[tuple, list[dict]] = {}
    for r in parallel_ok:
        groups.setdefault((r["clients"], r["kernels"]), []).append(r)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, ((clients, kernels), rows) in enumerate(sorted(groups.items())):
        color = colors[i % len(colors)]
        rows = sorted(rows, key=lambda r: r["threads"])
        threads = [r["threads"] for r in rows]
        seconds = [r["seconds"] for r in rows]
        marker = "o" if all(r["all_match"] for r in rows) else "x"
        ax.plot(threads, seconds, marker=marker, color=color,
                label=f"parallel: clients={clients}, kernels={kernels}")
        baseline = classic_ok.get((clients, kernels))
        if baseline is not None:
            ax.axhline(baseline, color=color, linestyle="--", alpha=0.5,
                       label=f"classic: clients={clients}, kernels={kernels}")
    ax.set_xlabel("NUMBA_NUM_THREADS")
    ax.set_ylabel("Wall-clock seconds (full UCR archive)")
    ax.set_title("Fed2RC parallel sweep: threads vs. runtime\n"
                  "(o = accuracy matched classic, x = mismatch, dashed = classic baseline)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_results.png", dpi=150)
    fig.savefig(out_dir / "sweep_results.svg")
    plt.close(fig)
    print(f"Wrote {out_dir / 'sweep_results.png'} and .svg")


def summarize(runs: list[dict]) -> bool:
    ok = [r for r in runs if r["returncode"] == 0]
    failed = [r for r in runs if r["returncode"] != 0]
    parallel_ok = [r for r in ok if r["engine"] == "parallel"]
    classic_seconds = {(r["clients"], r["kernels"]): r["seconds"]
                        for r in ok if r["engine"] == "classic"}
    accuracy_ok = bool(parallel_ok) and all(r["all_match"] for r in parallel_ok)

    groups: dict[tuple, list[dict]] = {}
    for r in parallel_ok:
        groups.setdefault((r["clients"], r["kernels"]), []).append(r)

    speedup_with_threads_ok = True
    beats_classic_ok = True
    for (clients, kernels), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["threads"])
        classic_s = classic_seconds.get((clients, kernels))
        fastest = min(r["seconds"] for r in rows)
        print(f"  clients={clients} kernels={kernels}: classic={classic_s:.1f}s"
              if classic_s is not None else f"  clients={clients} kernels={kernels}: classic=?")
        for r in rows:
            speedup = (classic_s / r["seconds"]) if classic_s else float("nan")
            print(f"    threads={r['threads']}: {r['seconds']:.1f}s "
                  f"(speedup {speedup:.2f}x)  match={r['all_match']}")
        if len(rows) >= 2:
            speedup_with_threads_ok &= rows[-1]["seconds"] <= rows[0]["seconds"]
        if classic_s is not None:
            beats_classic_ok &= fastest <= classic_s

    print(f"\nCompleted: {len(ok)}/{len(runs)}  Failed: {len(failed)}")
    print(f"ACCURACY MATCHES CLASSIC: {accuracy_ok}")
    print(f"SPEEDUP WITH THREADS: {speedup_with_threads_ok}")
    print(f"BEATS CLASSIC: {beats_classic_ok}")
    overall = accuracy_ok and speedup_with_threads_ok and beats_classic_ok
    print(f"PARALLEL SPEEDUP WITH MATCHING ACCURACY: {overall}")
    return overall


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clients-list", type=_int_list, default=[2, 4])
    parser.add_argument("--threads-list", type=_int_list, default=[1, 2, 4])
    parser.add_argument("--kernels-list", type=_int_list, default=[100])
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true",
                        help="Re-run configs even if already recorded in the JSON state")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned grid and commands, run nothing")
    args = parser.parse_args()

    pairs = list(itertools.product(args.clients_list, args.kernels_list))
    n_runs = len(pairs) * (1 + len(args.threads_list))
    print(f"Planned grid: {len(pairs)} (clients, kernels) pairs x "
          f"(1 classic baseline + {len(args.threads_list)} parallel thread counts) "
          f"= {n_runs} full-UCR-archive runs")
    for clients, kernels in pairs:
        print(f"  clients={clients} kernels={kernels} rounds={args.rounds}: "
              f"classic once, parallel at threads={args.threads_list}")
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
        "args": {"clients_list": args.clients_list, "threads_list": args.threads_list,
                  "kernels_list": args.kernels_list, "rounds": args.rounds, "tol": args.tol,
                  "out_dir": str(args.out_dir), "python": args.python},
    }

    for clients, kernels in pairs:
        run_group(clients, kernels, args.threads_list, args.rounds, args.tol,
                  args.out_dir, args.python, state, csv_path, json_path, args.force)

    plot_results(state["runs"], args.out_dir)
    summarize(state["runs"])


if __name__ == "__main__":
    main()
