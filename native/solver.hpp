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

// Ogni worker usa il proprio spazio di lavoro: matrice, pesi, predizioni e pivot.
// Non inizializziamo qui gli elementi: sara' il worker a scriverli per primo.
// Questo "first touch" puo' favorire la localita' NUMA se i thread sono legati
// ai core; la collocazione fisica delle pagine dipende dal sistema.
struct LambdaWorkspace {
    std::unique_ptr<double[]> matrix, weights, scores;
    std::unique_ptr<lapack_int[]> pivots;

    LambdaWorkspace(std::size_t d, std::size_t classes, std::size_t tile_rows)
        : matrix(new double[d * d]), weights(new double[d * classes]),
          scores(new double[tile_rows * classes]), pivots(new lapack_int[d]) {}
};

// Per ogni lambda risolviamo (A + lambda*I) W = b e valutiamo il classificatore.
// I candidati sono indipendenti: possiamo assegnarli a worker diversi (map).
// A e b sono per colonne, come richiesto qui da LAPACK; X e' per righe.
// Il chiamante valida gli input e limita le BLAS a un thread: il parallelismo
// e' gia' tra candidati, quindi evitiamo team annidati e oversubscription.
// status[i] vale 0 se tutto va bene, un errore LAPACK oppure -1000 per NaN/inf.
inline void lambda_map(const double* A, const double* b, const double* X,
                       const std::int64_t* y, const double* lambdas,
                       std::int64_t count, int d, int classes, int samples,
                       int threads, double* accuracies, int* status) {
    constexpr int tile_rows = 64;
    const std::size_t matrix_size = static_cast<std::size_t>(d) * d;
    const std::size_t weights_size = static_cast<std::size_t>(d) * classes;

    // Allochiamo prima della regione parallela: un errore di allocazione puo'
    // cosi' propagarsi al chiamante senza uscire da un worker OpenMP.
    // Ogni worker riusa i propri buffer per tutti i lambda assegnati.
    // Spazio per worker: O(d^2 + d*C + 64*C), con d feature e C classi.
    // Quindi aumentare i worker aumenta anche la memoria temporanea richiesta.
    std::vector<LambdaWorkspace> scratch;
    scratch.reserve(threads);
    for (int t = 0; t < threads; ++t)
        scratch.emplace_back(d, classes, tile_rows);

    #pragma omp parallel default(none) num_threads(threads) \
        shared(A, b, X, y, lambdas, count, d, classes, samples, accuracies, \
               status, scratch, matrix_size, weights_size, tile_rows)
    {
        // scratch e' condiviso, ma ciascun thread accede solo al proprio slot.
        // work e' un riferimento locale: i buffer sono a uso esclusivo del worker.
        auto& work = scratch[omp_get_thread_num()];

        // for ripartisce i candidati nel team gia' creato da parallel.
        // schedule(static) assegna gruppi contigui: i sistemi hanno tutti le
        // stesse dimensioni e un costo simile, quindi e' una scelta semplice.
        #pragma omp for schedule(static)
        for (std::int64_t i = 0; i < count; ++i) {
            // LAPACK modifica matrice e termini noti durante la soluzione:
            // copiamo A e b nei buffer del worker per preservare gli input.
            std::copy_n(A, matrix_size, work.matrix.get());
            std::copy_n(b, weights_size, work.weights.get());
            int info = 0;
            // Aggiungere lambda alla diagonale equivale a costruire A + lambda*I.
            for (int j = 0; j < d; ++j) {
                double& diagonal = work.matrix[static_cast<std::size_t>(j) * d + j];
                diagonal += lambdas[i];
                if (!std::isfinite(diagonal)) info = -1000;
            }

            // Risolviamo il sistema con fattorizzazione LU e pivoting parziale;
            // non calcoliamo esplicitamente l'inversa. La soluzione W prende
            // il posto di b in work.weights. Il layout per colonne evita
            // trasposizioni; l'interfaccia _work evita controlli NaN ripetuti
            // su input gia' verificati nel binding.
            if (info == 0)
                info = LAPACKE_dgesv_work(LAPACK_COL_MAJOR, d, classes,
                                        work.matrix.get(), d, work.pivots.get(),
                                        work.weights.get(), d);

            // Manteniamo la normalizzazione per classe del classificatore
            // originale. Anche queste scritture riguardano solo i buffer locali.
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
            // Prediciamo su blocchi di 64 campioni e contiamo subito gli esiti
            // corretti, senza conservare una matrice di tutti gli N*C punteggi.
            for (int first = 0; first < samples && info == 0;) {
                const int rows = std::min(tile_rows, samples - first);
                // DGEMM calcola i punteggi X_blocco * W per tutte le classi.
                // Gli stessi dati di W (d*C per colonne) si possono leggere
                // come W^T (C*d per righe): CblasTrans evita una copia trasposta.
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
                        // Scegliamo la classe col punteggio massimo. Usare >
                        // conserva la prima classe in caso di parita', come NumPy.
                        if (score[c] > score[predicted])
                            predicted = c;
                    }
                    if (info == 0 && predicted == y[first + row])
                        ++correct;
                }
                first += rows;
            }

            // L'iterazione i scrive solo accuracies[i] e status[i]: nessuna
            // competizione per aggiornare un "migliore globale" nel ciclo.
            // Celle vicine possono condividere cache line, ma qui ogni cella
            // viene scritta una volta, dopo il calcolo, senza lock o atomiche.
            accuracies[i] = info == 0 ? static_cast<double>(correct) / samples
                                     : std::numeric_limits<double>::quiet_NaN();
            status[i] = info;
        }
    } // Tutti i worker hanno finito: Python puo' controllare gli stati e ridurre.
}

} // namespace fed2rc
