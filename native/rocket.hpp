#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace fed2rc {

// ROCKET arithmetic follows sktime 1.1.0; see THIRD_PARTY_NOTICES.md.
// X is row-major. Output is column-major: each kernel writes contiguous sample
// features, with no intermediate [kernel, sample, feature] tensor or transpose.
// Blocks of samples keep the same input rows hot while visiting several kernels.
// All output cells have exactly one owner; no locks/atomics are needed.
inline void rocket_map(const float* X, const float* weights, const float* biases,
                       const std::int32_t* lengths, const std::int32_t* dilations,
                       const std::int32_t* paddings, std::int64_t samples,
                       std::int64_t timepoints, std::int64_t kernels, bool ppv_only,
                       int threads, float* out) {
    constexpr std::int64_t row_tile = 32;
    constexpr std::int64_t seed_tile = 8;
    const std::int64_t row_blocks = (samples + row_tile - 1) / row_tile;
    const std::int64_t seed_blocks = (kernels + seed_tile - 1) / seed_tile;
    const int width = ppv_only ? 1 : 2;

    // Rectangular 2-D worksharing: small N or small K still exposes parallelism.
    // Static chunks avoid a shared work queue. Only tile boundaries can share
    // cache lines; no frequently updated shared accumulator exists.
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
                    std::int64_t positive = 0;
                    float maximum = -std::numeric_limits<float>::infinity();
                    for (std::int64_t start = -padding; start < end; ++start) {
                        float sum = biases[s];
                        // Split the interior from the padded boundary: the
                        // common interior path has no per-tap bounds check.
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
                    out[(s * width) * samples + row] =
                        static_cast<float>(double(positive) / double(positions));
                    if (width == 2) out[(s * width + 1) * samples + row] = maximum;
                }
            }
        }
    } // implicit barrier: every output tile is complete before returning
}

} // namespace fed2rc
