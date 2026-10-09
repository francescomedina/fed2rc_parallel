#!/usr/bin/env bash
# Full-UCR classic vs OpenMP sweep. Start it detached so it survives logout:
#   nohup ./runner.sh > /dev/null 2>&1 &
# Check progress at any time with ./sweep_status.sh
cd "$(dirname "$0")" || exit 1
mkdir -p results/openmp_sweep
python -u code/sweep_runner.py --clients-list 4 --threads-list 1,2,4 --kernels-list 100,1000 \
    2>&1 | tee -a results/openmp_sweep/sweep.out
