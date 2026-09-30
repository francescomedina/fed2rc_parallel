#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <cblas.h>
#include <lapacke.h>
#include <omp.h>

namespace fed2rc {

// No value initialization: each worker first-touches its own pages, improving
// NUMA locality when the OpenMP runtime binds threads to CPU cores.
struct LambdaWorkspace {
    std::unique_ptr<double[]> matrix, weights, scores;
    std::unique_ptr<lapack_int[]> pivots;

    LambdaWorkspace(std::size_t d, std::size_t classes, std::size_t tile_rows)
        : matrix(new double[d * d]), weights(new double[d * classes]),
          scores(new double[tile_rows * classes]), pivots(new lapack_int[d]) {}
};

// A and b are column-major; X is row-major. The caller validates all shapes,
// positive dimensions, allocation-size products and threads, and limits BLAS
// to one thread to avoid nested parallelism / oversubscription.
// status[i] is LAPACK's info, or -1000 for a non-finite numerical result.
inline void lambda_map(const double* A, const double* b, const double* X,
                       const std::int64_t* y, const double* lambdas,
                       std::int64_t count, int d, int classes, int samples,
                       int threads, double* accuracies, int* status) {
    constexpr int tile_rows = 64;
    const std::size_t matrix_size = static_cast<std::size_t>(d) * d;
    const std::size_t weights_size = static_cast<std::size_t>(d) * classes;

    // Allocation failures propagate before OpenMP starts: C++ exceptions must
    // never escape a worker. Scratch is O(d² + d*C + 64*C) per worker, not O(N*C),
    // and is reused across lambda values assigned to that worker.
    std::vector<LambdaWorkspace> scratch;
    scratch.reserve(threads);
    for (int t = 0; t < threads; ++t)
        scratch.emplace_back(d, classes, tile_rows);

    #pragma omp parallel default(none) num_threads(threads) \
        shared(A, b, X, y, lambdas, count, d, classes, samples, accuracies, \
               status, scratch, matrix_size, weights_size, tile_rows)
    {
        auto& work = scratch[omp_get_thread_num()];

        // Static contiguous chunks suit equal-size dense solves and avoid a
        // dynamic task queue. Each iteration owns exactly one output cell.
        #pragma omp for schedule(static)
        for (std::int64_t i = 0; i < count; ++i) {
            std::copy_n(A, matrix_size, work.matrix.get());
            std::copy_n(b, weights_size, work.weights.get());
            int info = 0;
            for (int j = 0; j < d; ++j) {
                double& diagonal = work.matrix[static_cast<std::size_t>(j) * d + j];
                diagonal += lambdas[i];
                if (!std::isfinite(diagonal)) info = -1000;
            }

            // Inputs were checked once by the binding. The column-major work
            // interface calls the same LU without rescanning A/b for NaNs or
            // allocating transpose buffers for every lambda.
            if (info == 0)
                info = LAPACKE_dgesv_work(LAPACK_COL_MAJOR, d, classes,
                                        work.matrix.get(), d, work.pivots.get(),
                                        work.weights.get(), d);

            // Preserve the original classifier's per-class normalization.
            // Both the factorization and these writes are thread-private.
            for (int c = 0; c < classes && info == 0; ++c) {
                double* column = work.weights.get() + static_cast<std::size_t>(c) * d;
                double squared_norm = 0.0;
                for (int j = 0; j < d; ++j)
                    squared_norm += column[j] * column[j];
                const double norm = std::sqrt(squared_norm) + 1e-10;
                if (!std::isfinite(norm)) {
                    info = -1000;
                    break;
                }
                for (int j = 0; j < d; ++j)
                    column[j] /= norm;
            }

            std::int64_t correct = 0;
            // A small prediction tile is consumed immediately, avoiding a full
            // N-by-C temporary and its write/read traffic for each lambda.
            for (int first = 0; first < samples && info == 0;) {
                const int rows = std::min(tile_rows, samples - first);
                // Column-major d-by-C W is also row-major C-by-d W^T.
                // Thus this DGEMM computes X_tile * W without transposing W.
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            rows, classes, d, 1.0,
                            X + static_cast<std::size_t>(first) * d, d,
                            work.weights.get(), d, 0.0,
                            work.scores.get(), classes);
                for (int row = 0; row < rows && info == 0; ++row) {
                    const double* score = work.scores.get()
                                        + static_cast<std::size_t>(row) * classes;
                    int predicted = 0;
                    for (int c = 0; c < classes; ++c) {
                        if (!std::isfinite(score[c])) {
                            info = -1000;
                            break;
                        }
                        // Strict comparison preserves NumPy's first-argmax tie.
                        if (score[c] > score[predicted])
                            predicted = c;
                    }
                    if (info == 0 && predicted == y[first + row])
                        ++correct;
                }
                first += rows;
            }

            // One final write per cell, after expensive work. Adjacent cells
            // may share cache lines at chunk boundaries, but there is no hot
            // shared counter, atomic update, or lock inside the map.
            accuracies[i] = info == 0 ? static_cast<double>(correct) / samples
                                     : std::numeric_limits<double>::quiet_NaN();
            status[i] = info;
        }
    } // Implicit barrier: the caller can now check status and reduce accuracies.
}

} // namespace fed2rc
