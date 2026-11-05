#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <omp.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <memory>
#include "rocket.hpp"
#include "solver.hpp"

namespace py = pybind11;
// Layout degli array NumPy: C = elementi contigui per riga, F = per colonna.
// ROCKET legge una serie per riga; LAPACK lavora su matrici per colonne.
template <typename T> using CArray = py::array_t<T, py::array::c_style>;
template <typename T> using FArray = py::array_t<T, py::array::f_style>;

// Salviamo la configurazione OpenMP all'importazione, prima di limitare le BLAS.
// Alcune build di OpenBLAS condividono lo stesso runtime OpenMP: limitarle a un
// thread puo' cambiare anche il valore predefinito del parallelismo esterno.
// num_threads(...) permette di richiedere comunque il team salvato qui.
// OMP_NUM_THREADS va impostata prima di avviare Python.
static const int configured_threads = omp_get_max_threads();

// Gli errori sugli input diventano ValueError Python prima del calcolo parallelo.
static void require(bool condition, const char* message) {
    if (!condition) throw py::value_error(message);
}

static int workers(int requested, std::int64_t jobs) {
    // threads=0 usa la configurazione salvata. Non chiediamo piu' worker dei
    // lavori disponibili: avere thread senza lavoro aggiungerebbe solo costo.
    require(requested >= 0, "threads must be nonnegative (0 uses OMP_NUM_THREADS)");
    return int(std::min<std::int64_t>(jobs, requested ? requested : configured_threads));
}

template <typename T, int Flags> static void finite(const py::array_t<T, Flags>& a) {
    // Prima di leggere tramite puntatori C++, verifichiamo l'allineamento
    // richiesto dal tipo e rifiutiamo valori NaN o infiniti.
    require(reinterpret_cast<std::uintptr_t>(a.data()) % alignof(T) == 0,
            "arrays must be aligned");
    for (py::ssize_t i = 0; i < a.size(); ++i)
        require(std::isfinite(a.data()[i]), "inputs must be finite");
}

// Ponte Python -> ROCKET: controlla i dati, prepara l'output e chiama rocket_map.
// La distribuzione delle convoluzioni tra i thread e' in rocket.hpp.
static py::array_t<float> transform(
    CArray<float> X, CArray<float> weights, CArray<float> biases,
    CArray<std::int32_t> lengths, CArray<std::int32_t> dilations,
    CArray<std::int32_t> paddings, bool ppv_only, int threads, int temporal_chunks,
    py::object worker_ids) {
    require(temporal_chunks >= 1, "temporal_chunks must be positive");
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
    // Un'unica matrice finale: righe = serie, colonne = feature dei filtri.
    // Gli stride la dispongono per colonne, senza una trasposizione successiva.
    py::array_t<float> out({n, k * width}, {py::ssize_t(sizeof(float)),
                                         py::ssize_t(sizeof(float)) * n});
    // Un lavoro corrisponde a un blocco di 32 serie x 8 filtri (vedi rocket.hpp).
    const auto max_scratch = std::numeric_limits<std::size_t>::max() / sizeof(std::int64_t);
    require(k == 0 || std::size_t(n) <= max_scratch / std::size_t(k) / std::size_t(temporal_chunks),
            "temporal workspace is too large");
    // Existing tile worksharing is preserved for temporal_chunks=1.
    const auto jobs = temporal_chunks == 1 ? ((n + 31) / 32) * ((k + 7) / 8)
                                          : n * k * temporal_chunks;
    const int p = workers(threads, std::max<py::ssize_t>(1, jobs));
    // Allocate before releasing the GIL/starting workers. No per-task allocation.
    const auto partials = temporal_chunks == 1 ? 0 : std::size_t(n) * k * temporal_chunks;
    // Every map task initializes its own slot before the reduction barrier.
    // Avoid zero-filling scratch that is immediately overwritten by workers.
    std::unique_ptr<std::int64_t[]> counts(partials ? new std::int64_t[partials] : nullptr);
    std::unique_ptr<float[]> maxima(partials ? new float[partials] : nullptr);
    std::int32_t* owners = nullptr;
    if (!worker_ids.is_none()) {
        require(temporal_chunks > 1 && py::isinstance<py::array>(worker_ids),
                "_worker_ids requires temporal splitting and a NumPy array");
        auto ids = py::reinterpret_borrow<py::array>(worker_ids);
        require(ids.dtype().is(py::dtype::of<std::int32_t>()) && ids.ndim() == 1
                && std::size_t(ids.size()) == partials && ids.writeable()
                && (ids.flags() & py::array::c_style), "invalid _worker_ids buffer");
        require(reinterpret_cast<std::uintptr_t>(ids.data()) % alignof(std::int32_t) == 0,
                "_worker_ids must be aligned");
        owners = static_cast<std::int32_t*>(ids.mutable_data());
    }
    const float *xp = X.data(), *wp = weights.data(), *bp = biases.data();
    const std::int32_t *lp = lengths.data(), *dp = dilations.data(), *pp = paddings.data();
    float* result = out.mutable_data();
    if (k) {
        // Rilasciare il GIL lascia proseguire altri thread Python; i worker
        // OpenMP eseguono solo C++, senza usare API Python o allocare output.
        // Gli oggetti proprietari degli array restano vivi durante la chiamata.
        // Il distruttore di release riacquisisce il GIL a fine blocco (RAII).
        // Chi chiama non deve modificare gli input mentre vengono letti.
        py::gil_scoped_release release;
        if (temporal_chunks == 1)
            fed2rc::rocket_map(xp, wp, bp, lp, dp, pp, n, t, k, ppv_only, p, result);
        else
            fed2rc::rocket_map_temporal(xp, wp, bp, lp, dp, pp, n, t, k, ppv_only,
                                       p, temporal_chunks, result, counts.get(), maxima.get(), owners);
    }
    return out;
}

// Ponte Python -> ricerca di lambda: restituisce uno score e uno stato per
// candidato. Il migliore verra' scelto in Python, dopo il calcolo di tutti.
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
        // Puntatori e output sono gia' pronti. lambda_map, in solver.hpp,
        // distribuisce i candidati indipendenti tra i thread OpenMP.
        py::gil_scoped_release release;
        fed2rc::lambda_map(ap, bp, xp, yp, lp, count, int(d), classes, samples, p, scores, errors);
    }
    return py::make_tuple(accuracies, status);
}

// pybind11 registra le funzioni chiamabili da Python: questo blocco definisce
// l'interfaccia del modulo, mentre i calcoli partono alle singole chiamate.
PYBIND11_MODULE(_fed2rc_openmp, m) {
    m.attr("default_threads") = configured_threads;
    m.doc() = "Fed2RC native OpenMP kernels; buffers must not be mutated during calls.";
    // noconvert() rifiuta conversioni implicite di tipo/layout nel binding.
    // openmp_utils.py prepara gli array compatibili prima di arrivare qui.
    m.def("transform", &transform, py::arg("X").noconvert(), py::arg("weights").noconvert(),
          py::arg("biases").noconvert(), py::arg("lengths").noconvert(),
          py::arg("dilations").noconvert(), py::arg("paddings").noconvert(),
          py::arg("ppv_only") = true, py::arg("threads") = 0,
          py::arg("temporal_chunks") = 1, py::arg("_worker_ids") = py::none());
    m.def("lambda_scores", &lambda_scores, py::arg("A").noconvert(), py::arg("b").noconvert(),
          py::arg("X").noconvert(), py::arg("y").noconvert(), py::arg("lambdas").noconvert(),
          py::arg("threads") = 0);
    // []() e' una funzione anonima C++: esegue una prova del team OpenMP.
    // Questa regione serve alla diagnostica, non ai calcoli di ROCKET/ridge.
    m.def("runtime_info", []() {
        int team_size = 1;
        {
            py::gil_scoped_release release;
            // default(none) rende esplicita la condivisione delle variabili.
            // team_size e' condivisa, ma il costrutto single consente a un solo
            // thread (non necessariamente il principale) di scriverla.
            #pragma omp parallel default(none) num_threads(configured_threads) shared(team_size)
            {
                #pragma omp single
                team_size = omp_get_num_threads();
            }
        }
        // La regione e' terminata e il GIL e' riacquisito: possiamo creare
        // il dizionario Python. Team richiesto e team effettivo sono distinti;
        // omp_get_max_threads() non misura il numero di core fisici.
        py::dict info;
        info["openmp"] = _OPENMP;
        info["default_threads"] = configured_threads;
        info["probe_team_size"] = team_size;
        info["runtime_max_threads"] = omp_get_max_threads();
        return info;
    });
}
