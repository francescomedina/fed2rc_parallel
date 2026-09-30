"""Build the native CPU kernels. Requires an OpenMP compiler and OpenBLAS."""
import os
import shlex
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Standard GCC on Linux needs only -fopenmp. Apple Clang needs libomp explicitly.
compile_args = ["-O3", "-ffp-contract=off"]  # preserve operation order; no fast-math
link_args = []
include_dirs = ["native"]
library_dirs = []
libraries = shlex.split(os.environ.get(
    "FED2RC_BLAS_LIBRARIES", "openblas" if sys.platform == "darwin" else "lapacke openblas"
))
blas_prefix = os.environ.get("OPENBLAS_ROOT")
omp_prefix = os.environ.get("LIBOMP_ROOT")
if sys.platform == "darwin":
    homebrew = Path("/opt/homebrew" if Path("/opt/homebrew").is_dir() else "/usr/local")
    blas_prefix = blas_prefix or str(homebrew / "opt/openblas")
    omp_prefix = omp_prefix or str(homebrew / "opt/libomp")
    compile_args += ["-Xpreprocessor", "-fopenmp"]
    libraries += ["omp"]
    include_dirs += [str(Path(omp_prefix) / "include")]
    library_dirs += [str(Path(omp_prefix) / "lib")]
else:
    compile_args += ["-fopenmp"]
    link_args += ["-fopenmp"]
if blas_prefix:
    include_dirs += [str(Path(blas_prefix) / "include")]
    library_dirs += [str(Path(blas_prefix) / "lib")]
# Resolve the selected libraries at runtime as well as at build time.
link_args += [f"-Wl,-rpath,{p}" for p in library_dirs]
compile_args += shlex.split(os.environ.get("FED2RC_CXXFLAGS", ""))

setup(
    name="fed2rc-openmp",
    version="0.1.0",
    py_modules=["openmp_utils"],
    package_dir={"": "code"},
    ext_modules=[Pybind11Extension(
        "_fed2rc_openmp", ["native/bindings.cpp"], cxx_std=17,
        include_dirs=include_dirs, library_dirs=library_dirs, libraries=libraries,
        extra_compile_args=compile_args, extra_link_args=link_args,
        depends=["native/rocket.hpp", "native/solver.hpp"],
    )],
    cmdclass={"build_ext": build_ext},
)
