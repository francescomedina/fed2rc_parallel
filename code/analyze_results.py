"""Compare a fed2rc_no_fluke.py --debug run against the paper's baseline CSVs.

Parses a log produced by:
    NUMBA_NUM_THREADS=<n> python fed2rc_no_fluke.py --clients <c> --kernels <k> --debug > run.log

averages Test Accuracy / Test F1 across the 5 experiment seeds for each
dataset, and compares the result to the precomputed non-parallel baseline in
../results/Fed2RC/{IID,Non-IID}/<k>kernels_results.csv.

Note on IID vs Non-IID: fed2rc_no_fluke.py's main() only ever calls
split_function(..., my_split='uniform') -- the IID-like stratified split --
except for 5 datasets (DiatomSizeReduction, Fungi, PigAirwayPressure,
PigArtPressure, PigCVP) which use 'random'. There is no flag in the current
script that reproduces the Non-IID baseline's split, so a straight run should
be compared against the IID baseline; Non-IID is offered here only for
reference/manual experiments where you've edited the split logic yourself.

Usage:
    python analyze_results.py --log parallel.txt --kernels 100
    python analyze_results.py --log parallel.txt --kernels 100 --split both
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
)


def parse_log(log_path: Path) -> pd.DataFrame:
    """Return one row per (dataset, seed) with the Test Accuracy/F1 it logged."""
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
        NumSeeds=("Seed", "nunique"),
    ).reset_index()
    agg[["Acc. STD", "F1 STD"]] = agg[["Acc. STD", "F1 STD"]].fillna(0.0)
    return agg


def load_baseline(kernels: int, split: str) -> pd.DataFrame:
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
    return df[["Dataset", "Accuracy", "Acc. STD", "F1", "F1 STD"]]


def compare_run(log_path: Path, kernels: int, split: str, tol: float) -> dict:
    """Parse `log_path`, compare against the `split` baseline for `kernels`.

    Returns a summary dict including "all_match": bool, suitable for a
    sweep runner to record as a pass/fail correctness signal.
    """
    runs = parse_log(log_path)
    parallel = aggregate_per_dataset(runs)
    baseline = load_baseline(kernels, split)

    merged = parallel.merge(
        baseline, on="Dataset", how="outer", suffixes=(" (parallel)", " (baseline)"),
        indicator=True,
    )
    both = merged[merged["_merge"] == "both"].copy()
    both["Acc Diff"] = (both["Accuracy (parallel)"] - both["Accuracy (baseline)"]).abs()
    both["F1 Diff"] = (both["F1 (parallel)"] - both["F1 (baseline)"]).abs()
    both["Match"] = (both["Acc Diff"] <= tol) & (both["F1 Diff"] <= tol)

    only_parallel = merged[merged["_merge"] == "left_only"]["Dataset"].tolist()
    only_baseline = merged[merged["_merge"] == "right_only"]["Dataset"].tolist()

    return {
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
    }


def _print_summary(kernels: int, split: str, result: dict, tol: float) -> None:
    print(f"\n=== {split} baseline, {kernels} kernels (tol={tol}) ===")
    print(f"Datasets compared: {result['n_compared']}  "
          f"matched: {result['n_matched']}  mismatched: {result['n_mismatched']}")
    print(f"Mean |Acc diff|: {result['mean_acc_diff']:.4f}  "
          f"Mean |F1 diff|: {result['mean_f1_diff']:.4f}  "
          f"Max |Acc diff|: {result['max_acc_diff']:.4f}")
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
        print(mismatched[cols].head(15).to_string(index=False))
    print(f"\nALL MATCH: {result['all_match']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=Path("parallel.txt"),
                        help="Path to the fed2rc_no_fluke.py --debug stdout log")
    parser.add_argument("-k", "--kernels", type=int, default=100, choices=[100, 1000, 10000],
                        help="Which baseline kernel-count table to compare against")
    parser.add_argument("--split", choices=["IID", "Non-IID", "both"], default="IID",
                        help="Baseline split to compare against (default: IID, see module docstring)")
    parser.add_argument("--tol", type=float, default=0.02,
                        help="Max allowed |accuracy diff| and |F1 diff| to count as a match")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional path to write the merged per-dataset comparison CSV")
    args = parser.parse_args()

    if not args.log.exists():
        sys.exit(f"Log file not found: {args.log}")

    splits = ["IID", "Non-IID"] if args.split == "both" else [args.split]
    overall_match = True
    for split in splits:
        result = compare_run(args.log, args.kernels, split, args.tol)
        _print_summary(args.kernels, split, result, args.tol)
        overall_match = overall_match and result["all_match"]
        if args.out:
            out_path = args.out if len(splits) == 1 else args.out.with_stem(f"{args.out.stem}_{split}")
            result["table"].to_csv(out_path, index=False)
            print(f"Wrote {out_path}")

    print(f"\nOVERALL MATCH ({'/'.join(splits)}): {overall_match}")
    sys.exit(0 if overall_match else 1)


if __name__ == "__main__":
    main()
