#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <omp.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include "rocket.hpp"
#include "solver.hpp"

namespace py = pybind11;
template <typename T> using CArray = py::array_t<T, py::array::c_style>;
template <typename T> using FArray = py::array_t<T, py::array::f_style>;

// Capture the outer team before threadpoolctl limits BLAS. An OpenMP build of
// OpenBLAS may implement its setter using omp_set_num_threads(1) on this SAME
// runtime. Explicit num_threads below preserves the outer team while nested
// library work inherits the single-thread policy. Set OMP_NUM_THREADS at startup.
static const int configured_threads = omp_get_max_threads();

static void require(bool condition, const char* message) {
    if (!condition) throw py::value_error(message);
}

static int workers(int requested, std::int64_t jobs) {
    require(requested >= 0, "threads must be nonnegative (0 uses OMP_NUM_THREADS)");
    return int(std::min<std::int64_t>(jobs, requested ? requested : configured_threads));
}

template <typename T, int Flags> static void finite(const py::array_t<T, Flags>& a) {
    require(reinterpret_cast<std::uintptr_t>(a.data()) % alignof(T) == 0,
            "arrays must be aligned");
    for (py::ssize_t i = 0; i < a.size(); ++i)
        require(std::isfinite(a.data()[i]), "inputs must be finite");
}

static py::array_t<float> transform(
    CArray<float> X, CArray<float> weights, CArray<float> biases,
    CArray<std::int32_t> lengths, CArray<std::int32_t> dilations,
    CArray<std::int32_t> paddings, bool ppv_only, int threads) {
    require(X.ndim() == 2 && X.shape(0) > 0 && X.shape(1) >= 2,
            "X must have shape (samples > 0, timepoints >= 2)");
    require(weights.ndim() == 2 && weights.shape(1) == 11, "weights must have 11 columns");
    const auto n = X.shape(0), t = X.shape(1), k = weights.shape(0);
    require(biases.ndim() == 1 && lengths.ndim() == 1 && dilations.ndim() == 1
            && paddings.ndim() == 1 && biases.size() == k && lengths.size() == k
            && dilations.size() == k && paddings.size() == k, "inconsistent kernel arrays");
    require(reinterpret_cast<std::uintptr_t>(lengths.data()) % alignof(std::int32_t) == 0
            && reinterpret_cast<std::uintptr_t>(dilations.data()) % alignof(std::int32_t) == 0
            && reinterpret_cast<std::uintptr_t>(paddings.data()) % alignof(std::int32_t) == 0,
            "kernel arrays must be aligned");
    for (py::ssize_t s = 0; s < k; ++s) {
        const auto len = lengths.data()[s], dil = dilations.data()[s], pad = paddings.data()[s];
        require(len > 0 && len <= 11 && dil >= 0 && pad >= 0
                && t + 2LL * pad - (len - 1LL) * dil > 0, "invalid kernel geometry");
    }
    finite(X); finite(weights); finite(biases);
    const py::ssize_t width = ppv_only ? 1 : 2;
    require(k <= std::numeric_limits<py::ssize_t>::max() / n / width / py::ssize_t(sizeof(float)),
            "feature matrix is too large");
    py::array_t<float> out({n, k * width}, {py::ssize_t(sizeof(float)),
                                         py::ssize_t(sizeof(float)) * n});
    const int p = workers(threads, std::max<py::ssize_t>(1, ((n + 31) / 32) * ((k + 7) / 8)));
    const float *xp = X.data(), *wp = weights.data(), *bp = biases.data();
    const std::int32_t *lp = lengths.data(), *dp = dilations.data(), *pp = paddings.data();
    float* result = out.mutable_data();
    if (k) {
        // No Python objects/API access, allocation, or callback inside OpenMP.
        // Array owners above stay alive until this scope reacquires the GIL.
        py::gil_scoped_release release;
        fed2rc::rocket_map(xp, wp, bp, lp, dp, pp, n, t, k, ppv_only, p, result);
    }
    return out;
}

static py::tuple lambda_scores(FArray<double> A, FArray<double> b, CArray<double> X,
                              CArray<std::int64_t> y, CArray<double> lambdas, int threads) {
    require(A.ndim() == 2 && A.shape(0) > 0 && A.shape(0) == A.shape(1), "A must be square");
    const auto d = A.shape(0);
    require(b.ndim() == 2 && b.shape(0) == d && b.shape(1) > 0, "b must have shape (d, classes)");
    require(X.ndim() == 2 && X.shape(0) > 0 && X.shape(1) == d, "X must have shape (samples, d)");
    require(y.ndim() == 1 && y.size() == X.shape(0), "y must match samples");
    require(lambdas.ndim() == 1 && lambdas.size() > 0, "lambdas must be nonempty");
    require(d <= INT_MAX && b.shape(1) <= INT_MAX && X.shape(0) <= INT_MAX,
            "dimensions exceed supported BLAS integer range");
    const auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(double);
    require(std::size_t(d) <= max_elements / std::size_t(d)
            && std::size_t(d) <= max_elements / std::size_t(b.shape(1))
            && std::size_t(b.shape(1)) <= max_elements / 64,
            "solver workspace is too large");
    require(reinterpret_cast<std::uintptr_t>(y.data()) % alignof(std::int64_t) == 0,
            "y must be aligned");
    finite(A); finite(b); finite(X); finite(lambdas);
    for (py::ssize_t i = 0; i < lambdas.size(); ++i)
        require(lambdas.data()[i] > 0, "lambdas must be positive");
    for (py::ssize_t i = 0; i < y.size(); ++i)
        require(y.data()[i] >= 0 && y.data()[i] < b.shape(1), "class index out of range");
    py::array_t<double> accuracies(lambdas.size());
    py::array_t<int> status(lambdas.size());
    const int p = workers(threads, lambdas.size());
    const double *ap = A.data(), *bp = b.data(), *xp = X.data(), *lp = lambdas.data();
    const std::int64_t* yp = y.data();
    double* scores = accuracies.mutable_data();
    int* errors = status.mutable_data();
    const auto count = lambdas.size();
    const int classes = int(b.shape(1)), samples = int(X.shape(0));
    {
        py::gil_scoped_release release;
        fed2rc::lambda_map(ap, bp, xp, yp, lp, count, int(d), classes, samples, p, scores, errors);
    }
    return py::make_tuple(accuracies, status);
}

PYBIND11_MODULE(_fed2rc_openmp, m) {
    m.doc() = "Fed2RC native OpenMP kernels; buffers must not be mutated during calls.";
    m.def("transform", &transform, py::arg("X").noconvert(), py::arg("weights").noconvert(),
          py::arg("biases").noconvert(), py::arg("lengths").noconvert(),
          py::arg("dilations").noconvert(), py::arg("paddings").noconvert(),
          py::arg("ppv_only") = true, py::arg("threads") = 0);
    m.def("lambda_scores", &lambda_scores, py::arg("A").noconvert(), py::arg("b").noconvert(),
          py::arg("X").noconvert(), py::arg("y").noconvert(), py::arg("lambdas").noconvert(),
          py::arg("threads") = 0);
    m.def("runtime_info", []() {
        int team_size = 1;
        {
            py::gil_scoped_release release;
            #pragma omp parallel default(none) num_threads(configured_threads) shared(team_size)
            {
                #pragma omp single
                team_size = omp_get_num_threads();
            }
        }
        py::dict info;
        info["openmp"] = _OPENMP;
        info["default_threads"] = configured_threads;
        info["probe_team_size"] = team_size;
        info["runtime_max_threads"] = omp_get_max_threads();
        return info;
    });
}
