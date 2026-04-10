#pragma once

#include <cstddef>
#include <vector>

namespace pgkl {

    void matmul_tiled_cpu(const std::vector<float>& A,
                          const std::vector<float>& B,
                          std::vector<float>& C,
                          std::size_t M,
                          std::size_t N,
                          std::size_t K,
                          std::size_t tile_size = 32);

} // namespace pgkl
