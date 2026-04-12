#include "pgkl/matmul_tiled.hpp"

#include <algorithm>
#include <stdexcept>

namespace pgkl {

void matmul_tiled_cpu(const std::span<const float> a,
                      const std::span<const float> b,
                      std::span<float> c,
                      const std::size_t m,
                      const std::size_t n,
                      const std::size_t k,
                      const std::size_t tile_size) {
    if (tile_size == 0U) {
        throw std::invalid_argument("matmul_tiled_cpu: tile_size must be greater than zero");
    }
    if (a.size() != (m * k)) {
        throw std::invalid_argument("matmul_tiled_cpu: a.size() must equal m * k");
    }
    if (b.size() != (k * n)) {
        throw std::invalid_argument("matmul_tiled_cpu: b.size() must equal k * n");
    }
    if (c.size() != (m * n)) {
        throw std::invalid_argument("matmul_tiled_cpu: c.size() must equal m * n");
    }

    std::fill(c.begin(), c.end(), 0.0F);

    for (std::size_t row_block = 0; row_block < m; row_block += tile_size) {
        for (std::size_t k_block = 0; k_block < k; k_block += tile_size) {
            for (std::size_t col_block = 0; col_block < n; col_block += tile_size) {
                const auto row_end = std::min(row_block + tile_size, m);
                const auto k_end = std::min(k_block + tile_size, k);
                const auto col_end = std::min(col_block + tile_size, n);

                for (std::size_t row = row_block; row < row_end; ++row) {
                    for (std::size_t depth = k_block; depth < k_end; ++depth) {
                        const auto a_value = a[(row * k) + depth];
                        for (std::size_t col = col_block; col < col_end; ++col) {
                            c[(row * n) + col] += a_value * b[(depth * n) + col];
                        }
                    }
                }
            }
        }
    }
}

}  // namespace pgkl
