#include "pgkl/reduction.hpp"

#include "cuda_utils.cuh"

#include <cuda_runtime.h>

#include <span>

namespace pgkl {
namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void reduction_kernel(const float* input, float* partial_sums, const int count) {
    extern __shared__ float shared[];

    const int local_index = static_cast<int>(threadIdx.x);
    const int global_index = static_cast<int(blockIdx.x * blockDim.x + threadIdx.x);

    if (global_index < count) {
        shared[local_index] = input[global_index];
    } else {
        shared[local_index] = 0.0F;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            shared[local_index] += shared[local_index + stride];
        }
        __syncthreads();
    }

    if (local_index == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

} // namespace

auto reduction_cuda(const std::span<float> input) -> float {
    if (input.empty()) {
        return 0.0F;
    }

    float* device_input = nullptr;
    float* device_partial = nullptr;
    float result = 0.0F;

    try {
        cuda_detail::cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_input), sizeof(float) * input.size()),
                                "cudaMalloc reduction input");
        cuda_detail::cuda_check(
            cudaMemcpy(device_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice),
            "cudaMemcpy H2D reduction input");

        std::size_t current_count = input.size();
        float* current_input = device_input;
        bool current_input_needs_free = false;

        while (current_count > 1U) {
            const std::size_t block_count = cuda_detail::ceil_div(current_count, static_cast<std::size_t>(kThreadsPerBlock));

            cuda_detail::cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_partial), sizeof(float) * block_count),
                                    "cudaMalloc reduction partial sums");

            reduction_kernel<<<static_cast<unsigned int>(block_count), kThreadsPerBlock,
                               sizeof(float) * kThreadsPerBlock>>>(current_input,
                                                                    device_partial,
                                                                    static_cast<int>(current_count));

            cuda_detail::cuda_check(cudaGetLastError(), "reduction_kernel launch");
            cuda_detail::cuda_check(cudaDeviceSynchronize(), "reduction_kernel synchronize");

            if (current_input_needs_free) {
                cuda_detail::cuda_check(cudaFree(current_input), "cudaFree previous reduction buffer");
            }

            current_input = device_partial;
            current_input_needs_free = true;
            device_partial = nullptr;
            current_count = block_count;
        }

        cuda_detail::cuda_check(cudaMemcpy(&result, current_input, sizeof(float), cudaMemcpyDeviceToHost),
                                "cudaMemcpy D2H reduction result");

        if (current_input_needs_free) {
            cuda_detail::cuda_check(cudaFree(current_input), "cudaFree reduction final buffer");
            device_input = nullptr;
        }

        if (device_input != nullptr) {
            cuda_detail::cuda_check(cudaFree(device_input), "cudaFree reduction input");
        }
    } catch (...) {
        if (device_partial != nullptr) {
            static_cast<void>(cudaFree(device_partial));
        }
        if (device_input != nullptr) {
            static_cast<void>(cudaFree(device_input));
        }
        throw;
    }

    return result;
}

}  // namespace pgkl
