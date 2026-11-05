#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <omp.h>

namespace fed2rc {

// ROCKET trasforma ogni serie temporale in feature applicando filtri casuali.
// L'aritmetica segue sktime 1.1.0; vedere THIRD_PARTY_NOTICES.md.
// Qui "kernel" indica un filtro convoluzionale, gia' generato in Python.
// X e' memorizzata per righe; out per colonne, una feature su tutte le serie.
// Ogni coppia (serie, filtro) e' indipendente: e' una map parallela sui dati.
// Gli input sono condivisi in lettura e ogni cella di output ha un solo autore:
// per queste scritture non servono lock o operazioni atomiche.
inline void rocket_map(const float* X, const float* weights, const float* biases,
                       const std::int32_t* lengths, const std::int32_t* dilations,
                       const std::int32_t* paddings, std::int64_t samples,
                       std::int64_t timepoints, std::int64_t kernels, bool ppv_only,
                       int threads, float* out) {
    // Dividiamo il dominio in blocchi (tile) di 32 serie x 8 filtri.
    // Riusare le stesse serie con piu' filtri puo' favorire il riuso in cache.
    // Le divisioni arrotondate in alto includono anche l'ultimo blocco parziale.
    constexpr std::int64_t row_tile = 32;
    constexpr std::int64_t seed_tile = 8;
    const std::int64_t row_blocks = (samples + row_tile - 1) / row_tile;
    const std::int64_t seed_blocks = (kernels + seed_tile - 1) / seed_tile;
    const int width = ppv_only ? 1 : 2;

    // parallel for crea il team e divide le iterazioni tra i thread.
    // collapse(2) distribuisce le coppie (blocco di serie, blocco di filtri):
    // possiamo sfruttare piu' thread anche quando una dimensione e' piccola.
    // schedule(static) assegna il lavoro senza una coda dinamica di richieste.
    // shared indica i buffer comuni; firstprivate copia i parametri scalari
    // per ogni thread. default(none) richiede di esplicitare queste scelte.
    // Celle distinte possono comunque condividere una cache line ai confini:
    // possibile false sharing, pur senza una data race sulle celle.
    #pragma omp parallel for collapse(2) default(none) schedule(static) num_threads(threads) \
        shared(X, weights, biases, lengths, dilations, paddings, out) \
        firstprivate(samples, timepoints, kernels, row_blocks, seed_blocks, width)
    for (std::int64_t rb = 0; rb < row_blocks; ++rb) {
        for (std::int64_t sb = 0; sb < seed_blocks; ++sb) {
            const auto row_end = std::min(samples, (rb + 1) * row_tile);
            const auto seed_end = std::min(kernels, (sb + 1) * seed_tile);
            for (auto s = sb * seed_tile; s < seed_end; ++s) {
                const auto length = lengths[s];
                const auto dilation = dilations[s];
                const auto padding = paddings[s];
                const auto end = timepoints + padding - (length - 1LL) * dilation;
                const auto positions = end + padding;
                const auto* w = weights + s * 11;
                for (auto row = rb * row_tile; row < row_end; ++row) {
                    const auto* x = X + row * timepoints;
                    // Questi accumulatori sono locali all'iterazione: nessun
                    // altro thread aggiorna positive, maximum o sum.
                    std::int64_t positive = 0;
                    float maximum = -std::numeric_limits<float>::infinity();
                    // Il filtro scorre sulla serie: dilation e' la distanza
                    // tra i campioni letti; padding aggiunge zeri ai bordi.
                    for (std::int64_t start = -padding; start < end; ++start) {
                        float sum = biases[s];
                        // Se tutto il filtro e' dentro la serie, evitiamo un
                        // controllo dei limiti per ogni coefficiente. Ai bordi
                        // ignoriamo i campioni esterni, equivalenti a zeri.
                        if (start >= 0 && start + (length - 1LL) * dilation < timepoints) {
                            for (int tap = 0; tap < length; ++tap)
                                sum += w[tap] * x[start + std::int64_t(tap) * dilation];
                        } else {
                            for (int tap = 0; tap < length; ++tap) {
                                const auto pos = start + std::int64_t(tap) * dilation;
                                if (pos >= 0 && pos < timepoints) sum += w[tap] * x[pos];
                            }
                        }
                        // Consumiamo subito la risposta della convoluzione:
                        // non conserviamo un vettore di tutte le risposte.
                        positive += sum > 0.0f;
                        maximum = std::max(maximum, sum);
                    }
                    // PPV = numero di risposte positive / numero di posizioni.
                    // Se richiesto, la colonna seguente contiene il massimo.
                    out[(s * width) * samples + row] =
                        static_cast<float>(double(positive) / double(positions));
                    if (width == 2) out[(s * width + 1) * samples + row] = maximum;
                }
            }
        }
    } // Barriera implicita: tutti i blocchi sono completi prima del ritorno.
}

// Additional scheduling path. The original kernel/tile map above is unchanged.
// A job owns (kernel, sample, chunk). Split global OUTPUT indices q, not input:
// q in [0, T+2*pad-(length-1)*dilation), start=q-pad. Every original position
// belongs to exactly one half-open interval; input reads may overlap freely.
inline void rocket_map_temporal(
    const float* X, const float* weights, const float* biases,
    const std::int32_t* lengths, const std::int32_t* dilations,
    const std::int32_t* paddings, std::int64_t samples, std::int64_t timepoints,
    std::int64_t kernels, bool ppv_only, int threads, int chunks, float* out,
    std::int64_t* counts, float* maxima, std::int32_t* worker_ids = nullptr) {
    const auto pairs = samples * kernels;
    const auto tasks = pairs * chunks;
    const int width = ppv_only ? 1 : 2;
    #pragma omp parallel default(none) num_threads(threads) \
        shared(X, weights, biases, lengths, dilations, paddings, out, counts, maxima, worker_ids) \
        firstprivate(samples, timepoints, pairs, tasks, chunks, width)
    {
        #pragma omp for schedule(static)
        for (std::int64_t task = 0; task < tasks; ++task) {
            const auto chunk = task % chunks, pair = task / chunks;
            const auto row = pair % samples, s = pair / samples;
            const auto length = lengths[s], dilation = dilations[s], padding = paddings[s];
            const auto positions = timepoints + 2LL * padding - (length - 1LL) * dilation;
            const auto first = positions * chunk / chunks;
            const auto stop = positions * (chunk + 1) / chunks;
            const auto* x = X + row * timepoints;
            const auto* w = weights + s * 11;
            std::int64_t positive = 0;
            float maximum = -std::numeric_limits<float>::infinity();
            for (auto q = first; q < stop; ++q) {
                const auto start = q - padding;
                float sum = biases[s];
                // Same per-output arithmetic/order as rocket_map: no parallel
                // dot-product reduction and no padding at chunk boundaries.
                if (start >= 0 && start + (length - 1LL) * dilation < timepoints) {
                    for (int tap = 0; tap < length; ++tap)
                        sum += w[tap] * x[start + std::int64_t(tap) * dilation];
                } else {
                    for (int tap = 0; tap < length; ++tap) {
                        const auto pos = start + std::int64_t(tap) * dilation;
                        if (pos >= 0 && pos < timepoints) sum += w[tap] * x[pos];
                    }
                }
                positive += sum > 0.0f;
                maximum = std::max(maximum, sum);
            }
            counts[task] = positive;
            maxima[task] = maximum;
            if (worker_ids) worker_ids[task] = omp_get_thread_num();
        } // Barrier: all partials must exist before any feature is reduced.

        #pragma omp for schedule(static)
        for (std::int64_t pair = 0; pair < pairs; ++pair) {
            const auto row = pair % samples, s = pair / samples;
            const auto positions = timepoints + 2LL * paddings[s]
                                   - (lengths[s] - 1LL) * dilations[s];
            std::int64_t positive = 0;
            float maximum = -std::numeric_limits<float>::infinity();
            for (int chunk = 0; chunk < chunks; ++chunk) {
                positive += counts[pair * chunks + chunk];
                maximum = std::max(maximum, maxima[pair * chunks + chunk]);
            }
            out[(s * width) * samples + row] =
                static_cast<float>(double(positive) / double(positions));
            if (width == 2) out[(s * width + 1) * samples + row] = maximum;
        }
    }
}

} // namespace fed2rc
