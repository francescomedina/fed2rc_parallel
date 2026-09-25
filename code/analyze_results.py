"""Compare a fed2rc_no_fluke.py --debug run against a baseline, and report timing.

Parses logs produced by:
    NUMBA_NUM_THREADS=<n> python fed2rc_no_fluke.py --clients <c> --kernels <k> --engine parallel --debug > parallel.log
    python fed2rc_no_fluke.py --clients <c> --kernels <k> --engine classic --debug > classic.log

averages Test Accuracy / Test F1 / Time across the 5 experiment seeds for
each dataset. Two comparison sources are supported:

  --source classic (default): compare a --engine parallel log directly
      against a fresh --engine classic log of the SAME clients/kernels/rounds
      config. This is the live, apples-to-apples check -- same environment,
      same sklearn/sktime/numpy versions, same split, just a different ROCKET
      transform implementation -- and it also gives a real speedup number
      (classic seconds / parallel seconds) instead of just a correctness
      flag.

  --source csv: compare against the paper's precomputed, static baseline in
      ../results/Fed2RC/{IID,Non-IID}/<k>kernels_results.csv. No timing is
      available from these files. Kept for reference against the published
      numbers, but note fed2rc_no_fluke.py's split only reproduces the IID
      baseline (see below).

Note on IID vs Non-IID (--source csv only): fed2rc_no_fluke.py's main() only
ever calls split_function(..., my_split='uniform') -- the IID-like
stratified split -- except for 5 datasets (DiatomSizeReduction, Fungi,
PigAirwayPressure, PigArtPressure, PigCVP) which use 'random'. There is no
flag in the current script that reproduces the Non-IID baseline's split, so
compare against IID unless you've edited the split logic yourself.

Usage:
    python analyze_results.py --log parallel.log --classic-log classic.log
    python analyze_results.py --log parallel.log --source csv --kernels 100
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "Fed2RC"

DATASET_RE = re.compile(r"^DATASET\s+(.+?)\s*$")
SEED_RE = re.compile(r"^SEED\s+(\d+)\s*$")
RESULT_RE = re.compile(
    r"Test Accuracy:\s*([0-9.]+),\s*Test F1:\s*([0-9.eE+-]+)"
    r"(?:,\s*Time:\s*([0-9.eE+-]+)s)?"
)


def parse_log(log_path: Path) -> pd.DataFrame:
    """Return one row per (dataset, seed) with the Test Accuracy/F1/Time it logged.

    Time is NaN for logs written before the `--engine`/timing change.
    """
    dataset, seed = None, None
    rows = []
    with open(log_path) as f:
        for line in f:
            m = DATASET_RE.match(line)
            if m:
                dataset, seed = m.group(1), None
                continue
            m = SEED_RE.match(line)
            if m:
                seed = int(m.group(1))
                continue
            m = RESULT_RE.search(line)
            if m and dataset is not None and seed is not None:
                rows.append({
                    "Dataset": dataset,
                    "Seed": seed,
                    "Accuracy": float(m.group(1)),
                    "F1": float(m.group(2)),
                    "Time": float(m.group(3)) if m.group(3) is not None else np.nan,
                })
                seed = None  # each seed logs exactly one Test Accuracy line
    if not rows:
        raise ValueError(f"No 'Test Accuracy: ..., Test F1: ...' lines found in {log_path}")
    return pd.DataFrame(rows)


def aggregate_per_dataset(runs: pd.DataFrame) -> pd.DataFrame:
    agg = runs.groupby("Dataset").agg(
        Accuracy=("Accuracy", "mean"),
        **{"Acc. STD": ("Accuracy", "std")},
        F1=("F1", "mean"),
        **{"F1 STD": ("F1", "std")},
        Time=("Time", "mean"),
        **{"Time STD": ("Time", "std")},
        NumSeeds=("Seed", "nunique"),
    ).reset_index()
    agg[["Acc. STD", "F1 STD", "Time STD"]] = agg[["Acc. STD", "F1 STD", "Time STD"]].fillna(0.0)
    return agg


def load_baseline_csv(kernels: int, split: str) -> pd.DataFrame:
    path = RESULTS_DIR / split / f"{kernels}kernels_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"No baseline at {path}")
    df = pd.read_csv(path)
    # IID and Non-IID CSVs use slightly different column names; normalize.
    rename = {
        "Accuracy Mean": "Accuracy", "Accuracy Std": "Acc. STD",
        "F1 Mean": "F1", "F1 Std": "F1 STD",
    }
    df = df.rename(columns=rename)
    df["Time"] = np.nan
    df["Time STD"] = np.nan
    return df[["Dataset", "Accuracy", "Acc. STD", "F1", "F1 STD", "Time", "Time STD"]]


def _merge_and_score(parallel: pd.DataFrame, baseline: pd.DataFrame, tol: float) -> dict:
    merged = parallel.merge(
        baseline, on="Dataset", how="outer", suffixes=(" (parallel)", " (baseline)"),
        indicator=True,
    )
    both = merged[merged["_merge"] == "both"].copy()
    both["Acc Diff"] = (both["Accuracy (parallel)"] - both["Accuracy (baseline)"]).abs()
    both["F1 Diff"] = (both["F1 (parallel)"] - both["F1 (baseline)"]).abs()
    both["Match"] = (both["Acc Diff"] <= tol) & (both["F1 Diff"] <= tol)
    has_time = both["Time (parallel)"].notna().all() and both["Time (baseline)"].notna().all()
    if has_time:
        both["Speedup (baseline/parallel)"] = both["Time (baseline)"] / both["Time (parallel)"]

    only_parallel = merged[merged["_merge"] == "left_only"]["Dataset"].tolist()
    only_baseline = merged[merged["_merge"] == "right_only"]["Dataset"].tolist()

    result = {
        "table": both.drop(columns="_merge").sort_values("Acc Diff", ascending=False),
        "n_compared": len(both),
        "n_matched": int(both["Match"].sum()),
        "n_mismatched": int((~both["Match"]).sum()),
        "mean_acc_diff": float(both["Acc Diff"].mean()) if len(both) else float("nan"),
        "mean_f1_diff": float(both["F1 Diff"].mean()) if len(both) else float("nan"),
        "max_acc_diff": float(both["Acc Diff"].max()) if len(both) else float("nan"),
        "only_in_log": only_parallel,
        "only_in_baseline": only_baseline,
        "all_match": bool(len(both) > 0 and both["Match"].all()),
        "has_time": has_time,
    }
    if has_time:
        result["total_time_parallel"] = float(both["Time (parallel)"].sum())
        result["total_time_baseline"] = float(both["Time (baseline)"].sum())
        result["overall_speedup"] = result["total_time_baseline"] / result["total_time_parallel"]
    return result


def compare_run(log_path: Path, kernels: int, split: str, tol: float) -> dict:
    """Parse `log_path`, compare against the static CSV `split` baseline for `kernels`."""
    parallel = aggregate_per_dataset(parse_log(log_path))
    baseline = load_baseline_csv(kernels, split)
    return _merge_and_score(parallel, baseline, tol)


def compare_live(parallel_log: Path, classic_log: Path, tol: float) -> dict:
    """Parse both logs and compare the parallel engine directly against a
    freshly-run classic (pre-parallelization, serial) engine log -- same
    environment, same config, so this also yields a real speedup number.
    """
    parallel = aggregate_per_dataset(parse_log(parallel_log))
    classic = aggregate_per_dataset(parse_log(classic_log))
    return _merge_and_score(parallel, classic, tol)


def _print_summary(label: str, result: dict, tol: float) -> None:
    print(f"\n=== {label} (tol={tol}) ===")
    print(f"Datasets compared: {result['n_compared']}  "
          f"matched: {result['n_matched']}  mismatched: {result['n_mismatched']}")
    print(f"Mean |Acc diff|: {result['mean_acc_diff']:.4f}  "
          f"Mean |F1 diff|: {result['mean_f1_diff']:.4f}  "
          f"Max |Acc diff|: {result['max_acc_diff']:.4f}")
    if result["has_time"]:
        print(f"Total time -- baseline: {result['total_time_baseline']:.1f}s  "
              f"parallel: {result['total_time_parallel']:.1f}s  "
              f"overall speedup: {result['overall_speedup']:.2f}x")
    if result["only_in_log"]:
        print(f"In log but no baseline row: {result['only_in_log']}")
    if result["only_in_baseline"]:
        print(f"In baseline but missing from log "
              f"(dataset crashed or was skipped): {result['only_in_baseline']}")
    mismatched = result["table"][~result["table"]["Match"]]
    if len(mismatched):
        print("\nWorst mismatches:")
        cols = ["Dataset", "Accuracy (parallel)", "Accuracy (baseline)", "Acc Diff",
                "F1 (parallel)", "F1 (baseline)", "F1 Diff"]
        if result["has_time"]:
            cols += ["Time (parallel)", "Time (baseline)", "Speedup (baseline/parallel)"]
        print(mismatched[cols].head(15).to_string(index=False))
    print(f"\nALL MATCH: {result['all_match']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=Path("parallel.txt"),
                        help="Path to the --engine parallel --debug stdout log")
    parser.add_argument("--source", choices=["classic", "csv"], default="classic",
                        help="Compare against a live --engine classic log (default) "
                             "or the static baseline CSVs under results/Fed2RC")
    parser.add_argument("--classic-log", type=Path, default=None,
                        help="Path to the --engine classic --debug stdout log "
                             "(required when --source classic)")
    parser.add_argument("-k", "--kernels", type=int, default=100, choices=[100, 1000, 10000],
                        help="[--source csv] Which baseline kernel-count table to compare against")
    parser.add_argument("--split", choices=["IID", "Non-IID", "both"], default="IID",
                        help="[--source csv] Baseline split to compare against (default: IID, "
                             "see module docstring)")
    parser.add_argument("--tol", type=float, default=0.02,
                        help="Max allowed |accuracy diff| and |F1 diff| to count as a match")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional path to write the merged per-dataset comparison CSV")
    args = parser.parse_args()

    if not args.log.exists():
        sys.exit(f"Log file not found: {args.log}")

    overall_match = True
    if args.source == "classic":
        if args.classic_log is None:
            sys.exit("--source classic requires --classic-log <path>")
        if not args.classic_log.exists():
            sys.exit(f"Classic log file not found: {args.classic_log}")
        result = compare_live(args.log, args.classic_log, args.tol)
        _print_summary(f"parallel ({args.log}) vs classic ({args.classic_log})", result, args.tol)
        overall_match = result["all_match"]
        if args.out:
            result["table"].to_csv(args.out, index=False)
            print(f"Wrote {args.out}")
    else:
        splits = ["IID", "Non-IID"] if args.split == "both" else [args.split]
        for split in splits:
            result = compare_run(args.log, args.kernels, split, args.tol)
            _print_summary(f"{split} baseline, {args.kernels} kernels", result, args.tol)
            overall_match = overall_match and result["all_match"]
            if args.out:
                out_path = args.out if len(splits) == 1 else args.out.with_stem(f"{args.out.stem}_{split}")
                result["table"].to_csv(out_path, index=False)
                print(f"Wrote {out_path}")

    print(f"\nOVERALL MATCH: {overall_match}")
    sys.exit(0 if overall_match else 1)


if __name__ == "__main__":
    main()
