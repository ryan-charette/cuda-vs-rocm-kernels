#include "pgkl/matmul_tiled.hpp"

#include <algorithm>
#include <stdexcept>

namespace pgkl {

    void matmul_tiled_cpu(const std::vector<float>& A,
                          const std::vector<float>& B,
                          std::vector<float>& C,
                          std::size_t M,
                          std::size_t N,
                          std::size_t K,
                          std::size_t tile_size) {
        if (A.size() != M * K) {
            throw std::invalid_argument("matmul_tiled_cpu: A.size() must equal M * K");
        }
        if (B.size() != K * N) {
            throw std::invalid_argument("matmul_tiled_cpu: B.size() must equal K * N");
        }

        C.assign(M * N, 0.0f);

        for (std::size_t ii = 0; ii < M; ii += tile_size) {
            for (std::size_t kk = 0; kk < K; kk+= tile_size) {
                for (std::size_t jj = 0; jj < N; jj += tile_size) {
                    const std::size_t i_end = std::min(ii + tile_size, M);
                    const std::size_t k_end = std::min(kk + tile_size, K);
                    const std::size_t j_end = std::min(jj + tile_size, N);

                    for (std::size_t i = ii; i < i_end; i++) {
                        for (std::size_t k = kk; k < k_end; k++) {
                            const float a = A[i * K + k];
                            for (std::size_t j = jj; j < j_end; j++) {
                                C[i * N + j] += a * B[k * N + j];
                            }
                        }
                    }
                }
            }
        }
    } 

} // namespace pgkl
