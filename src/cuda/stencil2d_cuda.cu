#include "pgkl/stencil2d.hpp"

#include "cuda_utils.cuh"

#include <cuda_runtime.h>

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

    // Keep border cells unchanged so the stencil only touches interior points.
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

void stencil2d_cuda(const std::span<const float> input,
                    std::span<float> output,
                    const std::size_t rows,
                    const std::size_t cols) {
    if ((rows * cols) != input.size()) {
        throw std::invalid_argument("stencil2d_cuda: rows * cols must equal input.size()");
    }
    if (output.size() != input.size()) {
        throw std::invalid_argument("stencil2d_cuda: output.size() must equal input.size()");
    }

    float* device_input = nullptr;
    float* device_output = nullptr;

    try {
        cuda_detail::cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_input), sizeof(float) * input.size()),
                                "cudaMalloc stencil input");
        cuda_detail::cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(float) * output.size()),
                                "cudaMalloc stencil output");

        cuda_detail::cuda_check(
            cudaMemcpy(device_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice),
            "cudaMemcpy H2D stencil input");

        const dim3 threads{16U, 16U, 1U};
        const dim3 blocks{
            static_cast<unsigned int>(cuda_detail::ceil_div(cols, static_cast<std::size_t>(threads.x))),
            static_cast<unsigned int>(cuda_detail::ceil_div(rows, static_cast<std::size_t>(threads.y))),
            1U};

        stencil2d_kernel<<<blocks, threads>>>(device_input, device_output, rows, cols);
        cuda_detail::cuda_check(cudaGetLastError(), "stencil2d_kernel launch");
        cuda_detail::cuda_check(cudaDeviceSynchronize(), "stencil2d_kernel synchronize");

        cuda_detail::cuda_check(
            cudaMemcpy(output.data(), device_output, sizeof(float) * output.size(), cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H stencil output");

        cuda_detail::cuda_check(cudaFree(device_output), "cudaFree stencil output");
        cuda_detail::cuda_check(cudaFree(device_input), "cudaFree stencil input");
    } catch (...) {
        if (device_output != nullptr) {
            static_cast<void>(cudaFree(device_output));
        }
        if (device_input != nullptr) {
            static_cast<void>(cudaFree(device_input));
        }
        throw;
    }
}

}  // namespace pgkl
