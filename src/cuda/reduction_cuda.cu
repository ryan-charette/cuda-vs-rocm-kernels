#include "pgkl/reduction.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pgkl {

    void check_cuda(cudaError_t err, const std::string& message) {
        if (err != cudaSuccess) {
            throw std::runtime_error(message + ": " + cudaGetErrorString(err));
        }
    }

    __global__ void reduction_kernel(const float* input, float* partial_sums, int n) {
        extern __shared__ float shared_data[];

        int local_idx = threadIdx.x;
        int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (global_idx < n) {
            shared_data[local_idx] = input[global_idx];
        } else {
            shared_data[local_idx] = 0.0f;
        }

        __syncthreads();

        int stride = blockDim.x / 2;
        while (stride > 0) {
            if (local_idx < stride) {
                shared_data[local_idx] += shared_data[local_idx + stride];
            }
            __syncthreads();
            stride /= 2;
        }

        if (local_idx == 0) {
            partial_sums[blockIdx.x] = shared_data[0];
        }
    }

    float reduction_cuda(const std::vector<float>& input) {
        if (input.empty()) {
            return 0.0f;
        }

        const int threadsPerBlock = 256;

        float* d_input = nullptr;
        check_cuda(
            cudaMalloc((void**)&d_input, input.size() * sizeof(float)),
            "cudaMalloc d_input failed"
        );

        check_cuda(
            cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy host to device failed"
        );

        int current_size = input.size();
        float* d_current = d_input;

        while (current_size > 1) {
            int blocksPerGrid = (current_size + threadsPerBlock - 1) / threadsPerBlock;

            float* d_partial_sums = nullptr;
            check_cuda(
                cudaMalloc((void**)&d_partial_sums, blocksPerGrid * sizeof(float)),
                "cudaMalloc d_partial_sums failed"
            );

            reduction_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
                d_current, 
                d_partial_sums,
                current_size
            );

            check_cuda(cudaGetLastError(), "kernel launch failed");
            check_cuda(cudaDeviceSynchronize(), "kernel execution failed");

            cudaFree(d_current);
            d_current = d_partial_sums;
            current_size = blocksPerGrid;
        }

        float result = 0.0f;
        check_cuda(
            cudaMemcpy(&result, d_current, sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy device to host failed"
        );

        cudaFree(d_current);
        return result;
    }

    } // namespace pgkl
