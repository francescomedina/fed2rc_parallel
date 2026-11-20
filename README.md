<div align="center">

# Fed2RC: Federated Rocket Kernels and Ridge Classifier for Time Series Classification
Bruno Casella, Samuele Fonio, Lorenzo Sciandra, Claudio Gallicchio, Marco Aldinucci, Mirko Polato and Roberto Esposito

[![Conference](https://img.shields.io/badge/ECAI-2025-red)](https://ebooks.iospress.nl/doi/10.3233/FAIA251198)

</div>

# Overview

This repository contains the code to run and reproduce the experiments of the Fed2RC algorithm and baselines.

Please cite as:

```bibtex
@inproceedings{casella25fed2rc,
	title = {{Fed2RC}: {Federated} {Rocket} {Kernels} and {Ridge} {Classifier} for {Time} {Series} {Classification}},
	shorttitle = {{Fed2RC}},
	language = {en},
	booktitle = {{ECAI} 2025},
	publisher = {IOS Press},
	author = {Casella, Bruno and Fonio, Samuele and Sciandra, Lorenzo and Gallicchio, Claudio and Aldinucci, Marco and Polato, Mirko and Esposito, Roberto},
	year = {2025},
	doi = {10.3233/FAIA251198},
	pages = {3297--3304},
}
```

# Abstract
Time series classification is a pivotal task in modern machine learning, with widespread applications in fields such as healthcare, finance, and cybersecurity. While deep learning methods dominate recent developments, their resource demands and privacy limitations hinder deployment on low-power and decentralized environments. To address these challenges, we introduce Fed2RC, a fully federated and gradient-free approach that integrates the efficiency of Rocket-based feature extraction with the robustness of ridge regression in a privacy-preserving setting. Fed2RC builds upon two key ideas: (i) federated selection and aggregation of high-performing random convolution kernels, and (ii) incremental and communication-efficient updates of ridge classifier parameters using closed-form solutions. Additionally, we propose a novel federated protocol for selecting the global ridge regularization parameter $\lambda$, and show how to improve the communication efficiency by matrix factorization techniques. Extensive experiments on the UCR benchmark demonstrate that Fed2RC achieves state-of-the-art results with a fraction of the computation and communication costs.

## Usage
- Clone this repo.
- Download and unzip the UCRArchive Time Series Classification dataset: [UCRArchive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/), and place it in the same directory of the repository.
- Install the requirements. 
- Run experiments (Fed2RC and baselines) with the [fluke](https://makgyver.github.io/fluke/first_run.html) framework for FL. If you want to run Fed2RC experiments without fluke, please see `code/fed2rc_no_fluke.py`.

## Results
The results directory reports the accuracy and F1-score metrics for all datasets of the UCR Archive. Results are reported for both Fed2RC and baselines and expressed in terms of accuracy and f1-score. For Fed2RC, are reported also additional results on communication costs.


## Contributors
* Bruno Casella <bruno.casella@unito.it>  

* Samuele Fonio <samuele.fonio@unito.it>

* Lorenzo Sciandra <lorenzo.sciandra@unito.it>

* Claudio Gallicchio <claudio.gallicchio@unipi.it>

* Marco Aldinucci <marco.aldinucci@unito.it>

* Mirko Polato <mirko.polato@unito.it>

* Roberto Esposito <roberto.esposito@unito.it>

## Native OpenMP implementation

See [OPENMP.md](OPENMP.md) for build instructions, numerical regression tests,
and the shared-memory C++ implementation of ROCKET transforms and lambda search.

See [TEMPORAL.md](TEMPORAL.md) for optional intra-series temporal partitioning
in both Numba and C++/OpenMP, correctness tests, and reproducible benchmarks.

## Optional two-stage lambda solver

`two_stage` reduces the symmetric Gram matrix to band form and then to
tridiagonal form with parallel bulge chasing. It also transforms `b` and the
validation features, reusing the reduction for all lambda candidates. The
default LU solver and ROCKET transform are unchanged.

Requires CMake, an OpenMP C/C++ compiler (Homebrew LLVM on macOS), OpenBLAS
with LAPACKE, and pybind11. PLASMA is pinned and built locally under `build/`.

```sh
.venv/bin/python build_twostage.py
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests -q
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmark_twostage.py
cd code
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 ../.venv/bin/python fed2rc_no_fluke.py --engine parallel --lambda-solver two_stage --debug --datasets GunPoint
```

Python: `select_lambda(A, b, X, y, lambdas, threads=8, solver="two_stage")`.
Tile width is fixed at 64 and inner block size at 16. The method helps large
feature matrices with many lambda candidates; small workloads can be slower.
It uses more workspace than a one-stage reduction. Concurrent preparations are
serialized because PLASMA has a global context. Buffers must stay immutable
during calls. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.
