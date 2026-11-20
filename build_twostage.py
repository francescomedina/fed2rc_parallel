"""Build the optional PLASMA experiment; never changes the default backend.

Run with the project virtualenv. Requires CMake, git, a C/C++ OpenMP compiler,
OpenBLAS/LAPACKE and Python development headers. Dependencies stay in build/.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import pybind11

REVISION = "548e0bdc5a046a728770bccf14503c37732c80fc"
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plasma-source", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    base = ROOT / "build/two_stage"
    base.mkdir(parents=True, exist_ok=True)
    source = args.plasma_source.resolve() if args.plasma_source else base / "plasma"
    log = base / "build.log"

    def run(command):
        print(" ".join(map(str, command)), flush=True)
        with log.open("a") as output:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=output,
                                    stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text()[-6000:], file=sys.stderr)
            raise SystemExit(f"Build failed; see {log}")

    if not source.exists():
        run(["git", "clone", "https://github.com/icl-utk-edu/plasma.git", source])
        run(["git", "-C", source, "checkout", "--detach", REVISION])
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise SystemExit(f"Expected PLASMA {REVISION}, found {actual}")
    mac = sys.platform == "darwin"
    brew = Path("/opt/homebrew" if Path("/opt/homebrew").exists() else "/usr/local")
    blas = Path(os.environ.get("OPENBLAS_ROOT", str(brew / "opt/openblas") if mac else "/usr"))
    omp = Path(os.environ.get("LIBOMP_ROOT", str(brew / "opt/libomp") if mac else "/usr"))
    cc = os.environ.get("CC", str(brew / "opt/llvm/bin/clang") if mac else "gcc")
    cxx = os.environ.get("CXX", str(brew / "opt/llvm/bin/clang++") if mac else "g++")
    cmake = shutil.which("cmake") or str(brew / "opt/cmake/bin/cmake")
    install = base / "install"
    build = source / "build"
    omp_config = (["-DOpenMP_C_LIB_NAMES=omp", f"-DOpenMP_omp_LIBRARY={omp}/lib/libomp.dylib"]
                  if mac else [])
    run([cmake, "-S", source, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
         f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_PREFIX_PATH={blas};{omp}",
         "-DBLA_VENDOR=OpenBLAS", f"-DOpenBLAS_DIR={blas}/lib/cmake/openblas",
         *omp_config,
         f"-DPython_EXECUTABLE={sys.executable}", "-DBUILD_SHARED_LIBS=ON",
         f"-DCMAKE_INSTALL_PREFIX={install}"])
    run([cmake, "--build", build, "--target", "plasma", "-j", args.jobs])
    # Copy only runtime libraries and headers; upstream's install also requires
    # its unrelated plasmatest executable and omits generated precision headers.
    (install / "lib").mkdir(parents=True, exist_ok=True)
    (install / "include").mkdir(parents=True, exist_ok=True)
    for binary in build.glob("libplasma*"):
        if ".dylib" not in binary.name and ".so" not in binary.name:
            continue
        dest = install / "lib" / binary.name
        dest.unlink(missing_ok=True)
        if binary.is_symlink():
            dest.symlink_to(os.readlink(binary))
        else:
            shutil.copy2(binary, dest)
    if mac:
        # Upstream puts -fopenmp in global C flags: clang may auto-link its own
        # libomp despite FindOpenMP settings. Loading two runtimes corrupts team
        # state. Match the runtime already used by the main native extension.
        for name in ("libplasma.dylib", "libplasma_core_blas.dylib"):
            dylib = (install / "lib" / name).resolve()
            linked = subprocess.check_output(["otool", "-L", str(dylib)], text=True)
            for line in linked.splitlines()[1:]:
                dependency = line.strip().split(" (")[0]
                expected = dependency
                if "libomp" in dependency:
                    expected = str(omp / "lib/libomp.dylib")
                elif dependency.startswith("@rpath/libplasma_core_blas"):
                    expected = str(install / "lib" / Path(dependency).name)
                if dependency != expected:
                    run(["install_name_tool", "-change", dependency, expected, dylib])
    for header in (source / "include").glob("*.h"):
        shutil.copy2(header, install / "include" / header.name)
    inc = [f"-I{p}" for p in (install / "include", blas / "include", omp / "include",
                              pybind11.get_include(), sysconfig.get_paths()["include"])]
    common = ["-O3", "-ffp-contract=off", "-fopenmp", *inc]
    run([cc, *common, "-std=c11", "-fPIC", "-c", ROOT / "native/twostage_plasma.c",
         "-o", base / "bridge.o"])
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    dest = ROOT / "code" / ("_fed2rc_twostage" + suffix)
    link = [f"-L{install}/lib", f"-L{blas}/lib", f"-L{omp}/lib",
            f"-Wl,-rpath,{install}/lib", f"-Wl,-rpath,{blas}/lib", f"-Wl,-rpath,{omp}/lib",
            "-lplasma", "-lplasma_core_blas", "-lopenblas"]
    if mac:
        link += ["-lomp", "-undefined", "dynamic_lookup"]
    else:
        link += ["-llapacke"]
    run([cxx, *common, "-std=c++17", "-fPIC", "-shared", "-Inative",
         ROOT / "native/twostage_bindings.cpp", base / "bridge.o", *link, "-o", dest])
    print(f"Built {dest}\nPLASMA revision: {REVISION}\nBuild log: {log}")


if __name__ == "__main__":
    main()
