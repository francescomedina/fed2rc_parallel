#!/usr/bin/env bash
# Status of a code/sweep_runner.py sweep: done, failed, running, interrupted
# and pending configurations, with a rough estimate of the remaining time.
#
# Usage (from anywhere):
#   ./sweep_status.sh
#   ./sweep_status.sh --kernels-list 100,1000 --threads-list 1,2,4 --clients-list 4
#   ./sweep_status.sh --out-dir results/parallel_sweep
#
# The planned grid is taken from the command-line options if given, otherwise
# from the sweep's measurements.json metadata, otherwise from runner.sh's grid.
# Only the Python standard library is needed.
cd "$(dirname "$0")" || exit 1
exec python3 - "$@" <<'PYEOF'
import argparse, json, math, os, re, subprocess, sys, time
from pathlib import Path

N_DATASETS_DEFAULT = 128
ROOT = Path(".")  # the script cd's to the repo root; log_path values are relative to it

p = argparse.ArgumentParser()
p.add_argument("--out-dir", default="results/openmp_sweep")
p.add_argument("--clients-list")
p.add_argument("--threads-list")
p.add_argument("--kernels-list")
p.add_argument("--n-datasets", type=int, default=N_DATASETS_DEFAULT)
p.add_argument("--n-seeds", type=int, default=5)
a = p.parse_args()

out_dir = Path(a.out_dir)
json_path = out_dir / "measurements.json"
state = json.loads(json_path.read_text()) if json_path.exists() else {"metadata": None, "runs": []}
meta_args = (state.get("metadata") or {}).get("args", {})

def grid(cli, key, default):
    if cli:
        return [int(v) for v in cli.split(",") if v.strip()]
    return meta_args.get(key) or default

clients_list = grid(a.clients_list, "clients_list", [4])
threads_list = grid(a.threads_list, "threads_list", [1, 2, 4])
kernels_list = grid(a.kernels_list, "kernels_list", [100, 1000])
total_seeds = a.n_datasets * a.n_seeds

# Same order and log names as sweep_runner.run_group.
planned = []
for c in clients_list:
    for k in kernels_list:
        planned.append((c, k, "classic", ""))
        for t in threads_list:
            planned.append((c, k, "parallel", t))

def tag(c, k, engine, t):
    return f"c{c}_k{k}_{engine}" + (f"_t{t}" if t != "" else "")

def label(c, k, engine, t):
    return f"c{c} k{k} {engine}" + (f" t={t}" if t != "" else "")

def fmt(s):
    if s is None:
        return "?"
    s = int(round(s))
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s % 60:02d}s"

rows = {}
for r in state.get("runs", []):
    if "engine" not in r:
        continue  # rows written before the classic/parallel sweep format
    key = (r["clients"], r["kernels"], r["engine"], r["threads"])
    if key not in rows or r["returncode"] == 0 or rows[key]["returncode"] != 0:
        rows[key] = r

RESULT_RE = re.compile(r"Time:\s*([0-9.eE+-]+)s")
def parse_log(path):
    """Per-dataset summed seconds and seed count from a fed2rc_no_fluke log."""
    per_ds, n_seeds, ds, last_line, traceback = {}, 0, None, "", False
    if not path.exists():
        return None
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("DATASET "):
                ds = line.split(None, 1)[1].strip()
            m = RESULT_RE.search(line)
            if m and ds is not None:
                per_ds[ds] = per_ds.get(ds, 0.0) + float(m.group(1))
                n_seeds += 1
            if line.startswith("Traceback"):
                traceback = True
            if line.strip():
                last_line = line.strip()
    return {"per_ds": per_ds, "n_seeds": n_seeds, "current_ds": ds,
            "last_line": last_line, "traceback": traceback}

def running_processes():
    """Map planned keys to elapsed seconds for live fed2rc_no_fluke processes."""
    found = {}
    try:
        ps = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "pid=,etimes=,args="],
                            capture_output=True, text=True).stdout
    except OSError:
        return found
    for line in ps.splitlines():
        if "fed2rc_no_fluke.py" not in line or "--engine" not in line:
            continue
        pid, etimes, cmd = line.split(None, 2)
        def opt(name):
            m = re.search(rf"--{name}\s+(\S+)", cmd)
            return m.group(1) if m else None
        engine = opt("engine")
        threads = ""
        if engine == "parallel":
            try:
                env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
                for item in env:
                    if item.startswith(b"OMP_NUM_THREADS="):
                        threads = int(item.split(b"=", 1)[1])
            except OSError:
                pass
        try:
            key = (int(opt("clients")), int(opt("kernels")), engine, threads)
        except (TypeError, ValueError):
            continue
        found[key] = int(etimes)
    return found

live = running_processes()
logs = {key: parse_log(out_dir / "logs" / f"{tag(*key)}.log") for key in planned}

def reference_for(key):
    """Closest completed run (ref_key, ref_row, basis) with the same clients/kernels.

    Same engine is preferred (for parallel, the nearest thread count); otherwise
    the other engine, whose per-dataset times still weight progress well.
    """
    c, k, engine, t = key
    done = [(kk, rows[kk]) for kk in planned if kk in rows and rows[kk]["returncode"] == 0]
    same_engine = [x for x in done if x[0][:3] == (c, k, engine)]
    if same_engine:
        if engine == "parallel":
            same_engine.sort(key=lambda x: abs(x[0][3] - t))
        basis = "same config, other threads" if engine == "parallel" else "same config"
        return same_engine[0][0], same_engine[0][1], basis
    same_k = [x for x in done if x[0][:2] == (c, k)]
    if same_k:
        return same_k[0][0], same_k[0][1], "same kernels, other engine"
    return None, None, None

def estimate_total(key):
    """(seconds, basis) for a full run of `key`, or (None, None)."""
    ref_key, ref_row, _ = reference_for(key)
    if ref_row is not None and ref_key[2] == key[2]:
        return ref_row["seconds"], f"from {tag(*ref_key)}"
    return kernel_scaled_estimate(key)

def kernel_scaled_estimate(key):
    """Rough guess for a config whose kernel count has no completed run yet.

    Same clients/engine/threads at other kernel counts: with two or more, fit
    time ~ kernels**alpha through the two nearest (runtime grows far slower
    than linearly because seed selection shrinks the kernel set); with one,
    fall back to linear scaling, which overestimates.
    """
    c, k, engine, t = key
    done = sorted(((kk[1], rows[kk]["seconds"]) for kk in rows
                   if rows[kk]["returncode"] == 0 and kk[0] == c and kk[2] == engine
                   and kk[3] == t and kk[1] != k), key=lambda x: abs(math.log(x[0] / k)))
    if len(done) >= 2:
        (k1, s1), (k2, s2) = sorted(done[:2])
        alpha = math.log(s2 / s1) / math.log(k2 / k1)
        return s2 * (k / k2) ** alpha, f"rough: time ~ k^{alpha:.2f} from k={k1},{k2}"
    if len(done) == 1:
        k1, s1 = done[0]
        return s1 * k / k1, f"very rough: linear in kernels from k={k1}, likely high"
    return None, None

status_lines, remaining, unknown, counts = [], 0.0, 0, {}
for key in planned:
    c, k, engine, t = key
    row, log = rows.get(key), logs[key]
    if row is not None and row["returncode"] == 0:
        status = "DONE"
        detail = ""
        if engine == "parallel" and row.get("n_compared") not in ("", None):
            sp = row.get("overall_speedup")
            sp = f", speedup {sp:.2f}x" if isinstance(sp, (int, float)) else ""
            detail = f"match {row['n_matched']}/{row['n_compared']} ({row['all_match']}){sp}"
        status_lines.append((status, label(*key), fmt(row["seconds"]), detail))
    elif key in live:
        status = "RUNNING"
        elapsed = live[key]
        ref_key, ref_row, basis = reference_for(key)
        ref_log = parse_log(ROOT / ref_row["log_path"]) if ref_row else None
        done_ds = set(log["per_ds"]) if log else set()
        if log and log["current_ds"] in done_ds and log["n_seeds"] % a.n_seeds:
            done_ds.discard(log["current_ds"])  # current dataset still has seeds to go
        if ref_log and ref_log["per_ds"]:
            total_ref = sum(ref_log["per_ds"].values())
            frac = sum(v for d, v in ref_log["per_ds"].items() if d in done_ds) / total_ref
            how = f"weighted by {tag(*ref_key)} ({basis})"
        else:
            frac = (log["n_seeds"] if log else 0) / total_seeds
            how = "by finished seeds (no completed reference run yet)"
        if frac > 0:
            left = elapsed / frac - elapsed
            remaining += left
            eta = f"~{fmt(left)} left"
        else:
            unknown += 1
            eta = "too early to estimate"
        cur = log["current_ds"] if log else "?"
        status_lines.append((status, label(*key), fmt(elapsed),
                             f"{100 * frac:.{1 if frac < 0.1 else 0}f}% done, {eta}, "
                             f"now on {cur} [{how}]"))
    elif row is not None:
        status = "FAILED"
        last = log["last_line"] if log else "log missing"
        status_lines.append((status, label(*key), fmt(row["seconds"]),
                             f"rc={row['returncode']}: {last[:90]}  (see {row['log_path']})"))
    elif log is not None:
        status = "INTERRUPTED"
        cur = log["current_ds"] or "?"
        why = log["last_line"][:70] if log["traceback"] else f"stopped during {cur}"
        est, basis = estimate_total(key)
        rerun = f"; rerun ~{fmt(est)} ({basis})" if est is not None else "; it will rerun"
        status_lines.append((status, label(*key), "-",
                             f"partial log, no result recorded: {why}{rerun}"))
        if est is None:
            unknown += 1
        else:
            remaining += est
    else:
        status = "PENDING"
        est, basis = estimate_total(key)
        if est is None:
            unknown += 1
            status_lines.append((status, label(*key), "-", "no estimate yet"))
        else:
            remaining += est
            status_lines.append((status, label(*key), "-", f"~{fmt(est)} ({basis})"))
    counts[status] = counts.get(status, 0) + 1

print(f"Sweep status for {out_dir}  ({time.strftime('%Y-%m-%d %H:%M')})")
print(f"Grid: clients={clients_list} kernels={kernels_list} threads={threads_list} "
      f"({len(planned)} runs, classic first then threads, per clients/kernels pair)\n")
w = max(len(l[1]) for l in status_lines)
for status, lab, t, detail in status_lines:
    print(f"{status:<12}{lab:<{w + 2}}{t:>9}  {detail}")
print()
print("  ".join(f"{s}: {counts.get(s, 0)}" for s in
                ("DONE", "FAILED", "RUNNING", "INTERRUPTED", "PENDING")))
if not live and counts.get("PENDING", 0) + counts.get("INTERRUPTED", 0):
    print("No sweep process is running.")
left = counts.get("RUNNING", 0) + counts.get("INTERRUPTED", 0) + counts.get("PENDING", 0)
if left == 0:
    print("Nothing left to run.")
elif unknown == left:
    print("Estimated remaining: unknown until a first run of this grid completes.")
else:
    tail = f" + {unknown} run(s) with no estimate yet" if unknown else ""
    print(f"Estimated remaining: ~{fmt(remaining)}{tail}. Rough: dataset times vary a "
          f"lot, and extrapolating to new kernel counts is the least reliable part.")
PYEOF
