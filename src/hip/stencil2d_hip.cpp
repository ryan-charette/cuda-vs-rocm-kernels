#include "pgkl/stencil2d.hpp"

#include "hip_utils.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <span>
#include <stdexcept>

namespace pgkl {
namespace {

__global__ void stencil2d_kernel(const float* input,
                                 float* output,
                                 const std::size_t rows,
                                 const std::size_t cols) {
    const std::size_t col = (blockIdx.x * blockDim.x) + threadIdx.x;
    const std::size_t row = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (row >= rows || col >= cols) {
        return;
    }

    const std::size_t index = (row * cols) + col;

    if (row == 0U || col == 0U || row + 1U == rows || col + 1U == cols) {
        output[index] = input[index];
        return;
    }

    const float up = input[((row - 1U) * cols) + col];
    const float down = input[((row + 1U) * cols) + col];
    const float left = input[index - 1U];
    const float right = input[index + 1U];
    const float center = input[index];

    output[index] = up + down + left + right - (4.0F * center);
}

}  // namespace

void stencil2d_hip(const std::span<const float> input,
                   std::span<float> output,
                   const std::size_t rows,
                   const std::size_t cols) {
    if ((rows * cols) != input.size()) {
        throw std::invalid_argument("stencil2d_hip: rows * cols must equal input.size()");
    }
    if (output.size() != input.size()) {
        throw std::invalid_argument("stencil2d_hip: output.size() must equal input.size()");
    }

    float* device_input = nullptr;
    float* device_output = nullptr;

    try {
        hip_detail::hip_check(hipMalloc(reinterpret_cast<void**>(&device_input), sizeof(float) * input.size()),
                              "hipMalloc stencil input");
        hip_detail::hip_check(hipMalloc(reinterpret_cast<void**>(&device_output), sizeof(float) * output.size()),
                              "hipMalloc stencil output");

        hip_detail::hip_check(
            hipMemcpy(device_input, input.data(), sizeof(float) * input.size(), hipMemcpyHostToDevice),
            "hipMemcpy H2D stencil input");

        const dim3 threads{16U, 16U, 1U};
        const dim3 blocks{
            static_cast<unsigned int>(hip_detail::ceil_div(cols, static_cast<std::size_t>(threads.x))),
            static_cast<unsigned int>(hip_detail::ceil_div(rows, static_cast<std::size_t>(threads.y))),
            1U};

        hipLaunchKernelGGL(stencil2d_kernel, blocks, threads, 0, 0, device_input, device_output, rows, cols);
        hip_detail::hip_check(hipGetLastError(), "stencil2d_kernel launch");
        hip_detail::hip_check(hipDeviceSynchronize(), "stencil2d_kernel synchronize");

        hip_detail::hip_check(
            hipMemcpy(output.data(), device_output, sizeof(float) * output.size(), hipMemcpyDeviceToHost),
            "hipMemcpy D2H stencil output");

        hip_detail::hip_check(hipFree(device_output), "hipFree stencil output");
        hip_detail::hip_check(hipFree(device_input), "hipFree stencil input");
    } catch (...) {
        if (device_output != nullptr) {
            static_cast<void>(hipFree(device_output));
        }
        if (device_input != nullptr) {
            static_cast<void>(hipFree(device_input));
        }
        throw;
    }
}

}  // namespace pgkl
