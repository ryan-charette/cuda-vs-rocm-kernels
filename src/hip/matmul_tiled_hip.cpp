#include "pgkl/matmul_tiled.hpp"

#include "hip_utils.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <span>
#include <stdexcept>

namespace pgkl {
namespace {

__global__ void matmul_tiled_kernel_8(const float* a,
                                      const float* b,
                                      float* c,
                                      const std::size_t m,
                                      const std::size_t n,
                                      const std::size_t k) {
    __shared__ float tile_a[8][8];
    __shared__ float tile_b[8][8];

    const std::size_t row = (blockIdx.y * blockDim.y) + threadIdx.y;
    const std::size_t col = (blockIdx.x * blockDim.x) + threadIdx.x;
    const std::size_t local_row = threadIdx.y;
    const std::size_t local_col = threadIdx.x;

    float sum = 0.0F;
    const std::size_t tile_count = hip_detail::ceil_div(k, std::size_t{8});

    for (std::size_t tile = 0; tile < tile_count; ++tile) {
        const std::size_t tiled_col = (tile * 8U) + local_col;
        const std::size_t tiled_row = (tile * 8U) + local_row;

        tile_a[local_row][local_col] = (row < m && tiled_col < k) ? a[(row * k) + tiled_col] : 0.0F;
        tile_b[local_row][local_col] = (tiled_row < k && col < n) ? b[(tiled_row * n) + col] : 0.0F;
        __syncthreads();

        for (int inner = 0; inner < 8; ++inner) {
            sum += tile_a[local_row][inner] * tile_b[inner][local_col];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[(row * n) + col] = sum;
    }
}

__global__ void matmul_tiled_kernel_16(const float* a,
                                       const float* b,
                                       float* c,
                                       const std::size_t m,
                                       const std::size_t n,
                                       const std::size_t k) {
    __shared__ float tile_a[16][16];
    __shared__ float tile_b[16][16];

    const std::size_t row = (blockIdx.y * blockDim.y) + threadIdx.y;
    const std::size_t col = (blockIdx.x * blockDim.x) + threadIdx.x;
    const std::size_t local_row = threadIdx.y;
    const std::size_t local_col = threadIdx.x;

    float sum = 0.0F;
    const std::size_t tile_count = hip_detail::ceil_div(k, std::size_t{16});

    for (std::size_t tile = 0; tile < tile_count; ++tile) {
        const std::size_t tiled_col = (tile * 16U) + local_col;
        const std::size_t tiled_row = (tile * 16U) + local_row;

        tile_a[local_row][local_col] = (row < m && tiled_col < k) ? a[(row * k) + tiled_col] : 0.0F;
        tile_b[local_row][local_col] = (tiled_row < k && col < n) ? b[(tiled_row * n) + col] : 0.0F;
        __syncthreads();

        for (int inner = 0; inner < 16; ++inner) {
            sum += tile_a[local_row][inner] * tile_b[inner][local_col];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[(row * n) + col] = sum;
    }
}

__global__ void matmul_tiled_kernel_32(const float* a,
                                       const float* b,
                                       float* c,
                                       const std::size_t m,
                                       const std::size_t n,
                                       const std::size_t k) {
    __shared__ float tile_a[32][32];
    __shared__ float tile_b[32][32];

    const std::size_t row = (blockIdx.y * blockDim.y) + threadIdx.y;
    const std::size_t col = (blockIdx.x * blockDim.x) + threadIdx.x;
    const std::size_t local_row = threadIdx.y;
    const std::size_t local_col = threadIdx.x;

    float sum = 0.0F;
    const std::size_t tile_count = hip_detail::ceil_div(k, std::size_t{32});

    for (std::size_t tile = 0; tile < tile_count; ++tile) {
        const std::size_t tiled_col = (tile * 32U) + local_col;
        const std::size_t tiled_row = (tile * 32U) + local_row;

        tile_a[local_row][local_col] = (row < m && tiled_col < k) ? a[(row * k) + tiled_col] : 0.0F;
        tile_b[local_row][local_col] = (tiled_row < k && col < n) ? b[(tiled_row * n) + col] : 0.0F;
        __syncthreads();

        for (int inner = 0; inner < 32; ++inner) {
            sum += tile_a[local_row][inner] * tile_b[inner][local_col];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[(row * n) + col] = sum;
    }
}

void run_matmul_tiled_hip(const std::span<const float> a,
                          const std::span<const float> b,
                          std::span<float> c,
                          const std::size_t m,
                          const std::size_t n,
                          const std::size_t k,
                          const std::size_t tile_size) {
    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    try {
        hip_detail::hip_check(hipMalloc(reinterpret_cast<void**>(&device_a), sizeof(float) * a.size()),
                              "hipMalloc matmul A");
        hip_detail::hip_check(hipMalloc(reinterpret_cast<void**>(&device_b), sizeof(float) * b.size()),
                              "hipMalloc matmul B");
        hip_detail::hip_check(hipMalloc(reinterpret_cast<void**>(&device_c), sizeof(float) * c.size()),
                              "hipMalloc matmul C");

        hip_detail::hip_check(hipMemcpy(device_a, a.data(), sizeof(float) * a.size(), hipMemcpyHostToDevice),
                              "hipMemcpy H2D matmul A");
        hip_detail::hip_check(hipMemcpy(device_b, b.data(), sizeof(float) * b.size(), hipMemcpyHostToDevice),
                              "hipMemcpy H2D matmul B");

        const dim3 threads{static_cast<unsigned int>(tile_size), static_cast<unsigned int>(tile_size), 1U};
        const dim3 blocks{static_cast<unsigned int>(hip_detail::ceil_div(n, tile_size)),
                          static_cast<unsigned int>(hip_detail::ceil_div(m, tile_size)),
                          1U};

        if (tile_size == 8U) {
            hipLaunchKernelGGL(matmul_tiled_kernel_8, blocks, threads, 0, 0, device_a, device_b, device_c, m, n, k);
        } else if (tile_size == 16U) {
            hipLaunchKernelGGL(matmul_tiled_kernel_16, blocks, threads, 0, 0, device_a, device_b, device_c, m, n, k);
        } else {
            hipLaunchKernelGGL(matmul_tiled_kernel_32, blocks, threads, 0, 0, device_a, device_b, device_c, m, n, k);
        }

        hip_detail::hip_check(hipGetLastError(), "matmul_tiled kernel launch");
        hip_detail::hip_check(hipDeviceSynchronize(), "matmul_tiled kernel synchronize");

        hip_detail::hip_check(hipMemcpy(c.data(), device_c, sizeof(float) * c.size(), hipMemcpyDeviceToHost),
                              "hipMemcpy D2H matmul C");

        hip_detail::hip_check(hipFree(device_c), "hipFree matmul C");
        hip_detail::hip_check(hipFree(device_b), "hipFree matmul B");
        hip_detail::hip_check(hipFree(device_a), "hipFree matmul A");
    } catch (...) {
        if (device_c != nullptr) {
            static_cast<void>(hipFree(device_c));
        }
        if (device_b != nullptr) {
            static_cast<void>(hipFree(device_b));
        }
        if (device_a != nullptr) {
            static_cast<void>(hipFree(device_a));
        }
        throw;
    }
}

}  // namespace

void matmul_tiled_hip(const std::span<const float> a,
                      const std::span<const float> b,
                      std::span<float> c,
                      const std::size_t m,
                      const std::size_t n,
                      const std::size_t k,
                      const std::size_t tile_size) {
    if (tile_size == 0U) {
        throw std::invalid_argument("matmul_tiled_hip: tile_size must be greater than zero");
    }
    if (a.size() != (m * k)) {
        throw std::invalid_argument("matmul_tiled_hip: a.size() must equal m * k");
    }
    if (b.size() != (k * n)) {
        throw std::invalid_argument("matmul_tiled_hip: b.size() must equal k * n");
    }
    if (c.size() != (m * n)) {
        throw std::invalid_argument("matmul_tiled_hip: c.size() must equal m * n");
    }
    if (tile_size != 8U && tile_size != 16U && tile_size != 32U) {
        throw std::invalid_argument("matmul_tiled_hip: supported tile sizes are 8, 16, and 32");
    }

    run_matmul_tiled_hip(a, b, c, m, n, k, tile_size);
}

}  // namespace pgkl
