"""Measure lambda-count crossover of the CURRENT minimal backend against LU.

Run from repository root; no production files are changed. Results feed the
LaTeX document. JIT/imports/input construction and correctness checks are outside
the timing; ALL solver preparation, projections and scoring are inside.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

import numpy as np
from threadpoolctl import threadpool_info
from openmp_utils import select_lambda
import _fed2rc_twostage as backend

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", type=int, default=[900, 1024])
    parser.add_argument("--threads", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(*args.features, *args.threads, args.repeats) < 1:
        parser.error("positive sizes and repeats required")
    output = HERE / "dati"
    output.mkdir(exist_ok=True)
    records = []
    metadata = dict(cpu=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],text=True).strip()
                    if platform.system()=="Darwin" else platform.processor(),
                    platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                    plasma_revision=backend.plasma_revision, repeats=args.repeats,
                    samples=64, train_samples=256, classes=3, blas=threadpool_info(),
                    affinity="not pinned", scope="full API on prepared float64 inputs",
                    source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                      ["native/twostage_bindings.cpp","native/twostage_plasma.c","native/solver.hpp",
                       "native/bindings.cpp","code/openmp_utils.py","docs/two_stage/misura_soglia.py"]})
    def save(complete):
        (output/"misure.json").write_text(json.dumps(dict(metadata=metadata,complete=complete,records=records),indent=2))
        for d in args.features:
            for p in args.threads:
                subset=[r for r in records if r['d']==d and r['threads']==p]
                if not subset: continue
                with (output/f"d{d}_p{p}.csv").open("w") as f:
                    writer=csv.DictWriter(f,fieldnames=['L','lu_ms','two_ms','speedup','lu_q1','lu_q3','two_q1','two_q3'])
                    writer.writeheader()
                    for r in subset: writer.writerow({k:r[k] for k in writer.fieldnames})
    for d in args.features:
        rng=np.random.default_rng(2026+d)
        z=rng.uniform(0,1,size=(256,d))
        A=np.asfortranarray(z.T@z)
        b=np.asfortranarray(z.T@np.eye(3)[rng.integers(3,size=256)])
        X,y=rng.uniform(0,1,size=(64,d)),rng.integers(3,size=64)
        for p in args.threads:
            for L in [*range(1,33),64,100]:
                lambdas=np.logspace(-3,0,L)
                def call(method): return select_lambda(A,b,X,y,lambdas,threads=p,solver=method)
                expected=call('lu')
                def check(result):
                    assert result[:2]==expected[:2]
                    np.testing.assert_array_equal(result[2],expected[2])
                check(call('two_stage'))
                raw={'lu':[],'two_stage':[]}
                order=random.Random(173+d+p+L)
                for _ in range(args.repeats):
                    methods=['lu','two_stage'];order.shuffle(methods)
                    for method in methods:
                        start=perf_counter();result=call(method);elapsed=perf_counter()-start
                        raw[method].append(elapsed);check(result)
                lu,two=(statistics.median(raw[m])*1000 for m in ['lu','two_stage'])
                q={m:np.quantile(raw[m],[.25,.75])*1000 for m in raw}
                records.append(dict(d=d,threads=p,L=L,lu_ms=lu,two_ms=two,speedup=lu/two,
                    lu_q1=float(q['lu'][0]),lu_q3=float(q['lu'][1]),two_q1=float(q['two_stage'][0]),
                    two_q3=float(q['two_stage'][1]),raw_seconds=raw,accuracy=float(expected[1]),
                    selected_lambda=float(expected[0]),accuracy_sha256=hashlib.sha256(expected[2].tobytes()).hexdigest()))
                print(f"D={d} P={p} L={L:3d} LU={lu:.3f} ms TWO={two:.3f} ms S={lu/two:.3f}",flush=True)
                save(False)
    save(True)


if __name__=="__main__": main()
