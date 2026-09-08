// Linux: g++ -O3 -std=c++17 -fopenmp -ffp-contract=off benchmark_solver.cpp -o benchmark_solver
// macOS/Homebrew: g++-15 -O3 -std=c++17 -fopenmp -ffp-contract=off benchmark_solver.cpp -o benchmark_solver
// Run: ./benchmark_solver 96 100 4 5
// Same scalar float64 LU/pivoting and workload as benchmark_solver.py.
// Do not use -Ofast/-ffast-math. Generation, warmup and checks are not timed;
// allocations, normalization, classification and the final reduction are timed.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

using Matrix = std::vector<double>;
constexpr int classes = 3;

struct Problem {
    int n;
    Matrix A, b, X, lambdas;
    std::vector<int> y;
};

Problem make_problem(int n, int count) {
    Problem p{n, Matrix(std::size_t(n) * n), Matrix(std::size_t(n) * classes),
              Matrix(2 * std::size_t(n) * n), Matrix(count), std::vector<int>(2 * n)};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            p.A[std::size_t(i) * n + j] = i == j ? 2.0 + double(i) / n
                : ((i + j) % 7 - 3) / (20.0 * n);
        for (int c = 0; c < classes; ++c)
            p.b[i * classes + c] = ((i * 3 + c * 5) % 17 - 8) / 8.0;
    }
    for (int r = 0; r < 2 * n; ++r) {
        p.y[r] = r % classes;
        for (int j = 0; j < n; ++j)
            p.X[std::size_t(r) * n + j] = ((r * 7 + j * 3) % 19 - 9) / 9.0;
    }
    for (int i = 0; i < count; ++i)
        p.lambdas[i] = std::pow(10.0, -3.0 + 3.0 * i / (count - 1));
    return p;
}

Matrix lu_solve(const Problem& p, double lam) {
    const int n = p.n;
    Matrix lu = p.A, W = p.b;  // Task-private storage; no shared mutable scratch.
    auto a = [&](int i, int j) -> double& { return lu[std::size_t(i) * n + j]; };
    auto w = [&](int i, int c) -> double& { return W[i * classes + c]; };
    for (int i = 0; i < n; ++i) a(i, i) += lam;
    for (int k = 0; k < n; ++k) {
        int pivot = k;
        for (int i = k + 1; i < n; ++i)
            if (std::abs(a(i, k)) > std::abs(a(pivot, k))) pivot = i;
        // Generated matrices are strictly diagonally dominant SPD, so no zero
        // pivots occur. A failure is converted to a per-task NaN by search().
        if (a(pivot, k) == 0.0) throw std::runtime_error("Singular matrix");
        if (pivot != k) {
            for (int j = 0; j < n; ++j) std::swap(a(k, j), a(pivot, j));
            for (int c = 0; c < classes; ++c) std::swap(w(k, c), w(pivot, c));
        }
        for (int i = k + 1; i < n; ++i) {
            a(i, k) /= a(k, k);
            for (int j = k + 1; j < n; ++j) a(i, j) -= a(i, k) * a(k, j);
        }
    }
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < classes; ++c)
            for (int j = 0; j < i; ++j) w(i, c) -= a(i, j) * w(j, c);
    for (int i = n - 1; i >= 0; --i)
        for (int c = 0; c < classes; ++c) {
            for (int j = i + 1; j < n; ++j) w(i, c) -= a(i, j) * w(j, c);
            w(i, c) /= a(i, i);
        }
    return W;
}

double accuracy(const Problem& p, double lam) {
    Matrix W = lu_solve(p, lam);
    for (int c = 0; c < classes; ++c) {
        double norm = 0.0;
        for (int j = 0; j < p.n; ++j) norm += W[j * classes + c] * W[j * classes + c];
        norm = std::sqrt(norm) + 1e-10;
        for (int j = 0; j < p.n; ++j) W[j * classes + c] /= norm;
    }
    int correct = 0;
    for (int row = 0; row < 2 * p.n; ++row) {
        int best_class = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (int c = 0; c < classes; ++c) {
            double score = 0.0;
            for (int j = 0; j < p.n; ++j)
                score += p.X[std::size_t(row) * p.n + j] * W[j * classes + c];
            if (score > best_score) { best_class = c; best_score = score; }
        }
        correct += best_class == p.y[row];
    }
    return double(correct) / p.y.size();
}

struct Result { Matrix scores; int best; };

Result search(const Problem& p, int threads) {
    const int count = int(p.lambdas.size());
    Matrix scores(count);
    // Explicit ownership: p read-only, scores[i] written exactly once. Static
    // contiguous chunks fit equal-cost dense solves and limit shared boundaries.
    // Rare final stores may share lines; no hot shared counters or locks exist.
    #pragma omp parallel for default(none) shared(p, scores) firstprivate(count) num_threads(threads) schedule(static)
    for (int i = 0; i < count; ++i) {
        try { scores[i] = accuracy(p, p.lambdas[i]); }
        catch (...) { scores[i] = std::numeric_limits<double>::quiet_NaN(); }
    }
    // Implicit barrier above: a sequential O(L) lexicographic reduction also
    // preserves the largest-lambda tie rule without a custom OpenMP reduction.
    int best = 0;
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(scores[i])) throw std::runtime_error("Lambda solve failed");
        if (scores[i] > scores[best] ||
            (scores[i] == scores[best] && p.lambdas[i] > p.lambdas[best])) best = i;
    }
    return {std::move(scores), best};
}

double median(Matrix values) {
    std::sort(values.begin(), values.end());
    const auto k = values.size() / 2;
    return values.size() % 2 ? values[k] : (values[k - 1] + values[k]) / 2.0;
}

void print_array(const Matrix& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}

int main(int argc, char** argv) {
    try {
        if (argc > 5) throw std::invalid_argument("Usage: benchmark_solver [n count threads repeats]");
        const int n = argc > 1 ? std::stoi(argv[1]) : 96;
        const int count = argc > 2 ? std::stoi(argv[2]) : 100;
        const int threads = argc > 3 ? std::stoi(argv[3]) : 4;
        const int repeats = argc > 4 ? std::stoi(argv[4]) : 5;
        if (n < 1 || count < 2 || threads < 1 || repeats < 1)
            throw std::invalid_argument("Require n>=1, count>=2, threads>=1, repeats>=1");
        omp_set_dynamic(0);
        omp_set_max_active_levels(1);
        int actual_threads = 0;
        #pragma omp parallel num_threads(threads) default(none) shared(actual_threads)
        {
            #pragma omp single
            actual_threads = omp_get_num_threads();
        }
        if (actual_threads != threads)
            throw std::runtime_error("OpenMP runtime did not supply the requested thread count");
        const Problem p = make_problem(n, count);
        Result reference = search(p, 1), result = search(p, threads); // Warmup.
        Matrix times[2];
        for (int repetition = 0; repetition < repeats; ++repetition)
            for (int turn = 0; turn < 2; ++turn) {
                const int mode = repetition % 2 == 0 ? turn : 1 - turn;
                const auto start = std::chrono::steady_clock::now();
                Result current = search(p, mode == 0 ? 1 : threads);
                const auto stop = std::chrono::steady_clock::now();
                times[mode].push_back(std::chrono::duration<double>(stop - start).count());
                if (current.best != reference.best || current.scores != reference.scores)
                    throw std::runtime_error("Serial/parallel results differ");
                result = std::move(current);
            }
        const Matrix W = lu_solve(p, p.lambdas[result.best]);
        double residual2 = 0.0, anorm2 = 0.0, wnorm2 = 0.0, bnorm2 = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double a = p.A[std::size_t(i) * n + j] + (i == j ? p.lambdas[result.best] : 0.0);
                anorm2 += a * a;
            }
            for (int c = 0; c < classes; ++c) {
                double v = 0.0;
                for (int j = 0; j < n; ++j)
                    v += (p.A[std::size_t(i) * n + j] + (i == j ? p.lambdas[result.best] : 0.0))
                        * W[j * classes + c];
                v -= p.b[i * classes + c];
                residual2 += v * v;
                wnorm2 += W[i * classes + c] * W[i * classes + c];
                bnorm2 += p.b[i * classes + c] * p.b[i * classes + c];
            }
        }
        const double residual = std::sqrt(residual2) /
            (std::sqrt(anorm2) * std::sqrt(wnorm2) + std::sqrt(bnorm2));
        if (!std::isfinite(residual) || residual > 1e-12)
            throw std::runtime_error("LU residual exceeds tolerance");
        const double serial = median(times[0]), parallel = median(times[1]);
        std::cout << std::setprecision(17) << "{\"backend\":\"openmp\",\"n\":" << n
            << ",\"count\":" << count << ",\"threads\":" << threads << ",\"repeats\":" << repeats
            << ",\"serial_seconds\":" << serial << ",\"parallel_seconds\":" << parallel
            << ",\"speedup\":" << serial / parallel << ",\"best_index\":" << result.best
            << ",\"best_lambda\":" << p.lambdas[result.best]
            << ",\"accuracy\":" << result.scores[result.best] << ",\"residual\":" << residual
            << ",\"scores\":";
        print_array(result.scores);
        std::cout << ",\"solution\":";
        print_array(W);
        std::cout << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
