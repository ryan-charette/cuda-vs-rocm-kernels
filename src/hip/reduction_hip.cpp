#include "pgkl/reduction.hpp"

#include "hip_utils.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <span>

namespace pgkl {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::size_t kElementsPerThread = 2U;

__global__ void reduction_kernel(const float* __restrict__ input, float* __restrict__ partial_sums, const int count) {
    extern __shared__ float shared[];

    const int local_index = static_cast<int>(threadIdx.x);
    const int block_offset = static_cast<int>(blockIdx.x * blockDim.x * kElementsPerThread);
    const int first_index = block_offset + local_index;
    const int second_index = first_index + static_cast<int>(blockDim.x);

    float thread_sum = 0.0F;
    if (first_index < count) {
        thread_sum += input[first_index];
    }
    if (second_index < count) {
        thread_sum += input[second_index];
    }

    shared[local_index] = thread_sum;
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

}  // namespace

auto reduction_hip(const std::span<const float> input) -> float {
    if (input.empty()) {
        return 0.0F;
    }

    float* device_input = nullptr;
    float* device_partial = nullptr;
    float result = 0.0F;

    try {
        hip_detail::hip_check(hipMalloc(reinterpret_cast<void**>(&device_input), sizeof(float) * input.size()),
                              "hipMalloc reduction input");
        hip_detail::hip_check(
            hipMemcpy(device_input, input.data(), sizeof(float) * input.size(), hipMemcpyHostToDevice),
            "hipMemcpy H2D reduction input");

        std::size_t current_count = input.size();
        float* current_input = device_input;
        bool current_input_needs_free = false;

        while (current_count > 1U) {
            const std::size_t block_count =
                hip_detail::ceil_div(current_count, static_cast<std::size_t>(kThreadsPerBlock) * kElementsPerThread);

            hip_detail::hip_check(hipMalloc(reinterpret_cast<void**>(&device_partial), sizeof(float) * block_count),
                                  "hipMalloc reduction partial sums");

            hipLaunchKernelGGL(reduction_kernel,
                               dim3(static_cast<unsigned int>(block_count)),
                               dim3(kThreadsPerBlock),
                               sizeof(float) * kThreadsPerBlock,
                               0,
                               current_input,
                               device_partial,
                               static_cast<int>(current_count));

            hip_detail::hip_check(hipGetLastError(), "reduction_kernel launch");
            hip_detail::hip_check(hipDeviceSynchronize(), "reduction_kernel synchronize");

            if (current_input_needs_free) {
                hip_detail::hip_check(hipFree(current_input), "hipFree previous reduction buffer");
            }

            current_input = device_partial;
            current_input_needs_free = true;
            device_partial = nullptr;
            current_count = block_count;
        }

        hip_detail::hip_check(hipMemcpy(&result, current_input, sizeof(float), hipMemcpyDeviceToHost),
                              "hipMemcpy D2H reduction result");

        if (current_input_needs_free) {
            hip_detail::hip_check(hipFree(current_input), "hipFree reduction final buffer");
            device_input = nullptr;
        }

        if (device_input != nullptr) {
            hip_detail::hip_check(hipFree(device_input), "hipFree reduction input");
        }
    } catch (...) {
        if (device_partial != nullptr) {
            static_cast<void>(hipFree(device_partial));
        }
        if (device_input != nullptr) {
            static_cast<void>(hipFree(device_input));
        }
        throw;
    }

    return result;
}

}  // namespace pgkl
