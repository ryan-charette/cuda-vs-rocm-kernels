#pragma once

#include <cstddef>
#include <span>

namespace pgkl {

void matmul_tiled_cpu(std::span<const float> a,
                      std::span<const float> b,
                      std::span<float>& c,
                      std::size_t m,
                      std::size_t n,
                      std::size_t k,
                      std::size_t tile_size = 32);

} // namespace pgkl
