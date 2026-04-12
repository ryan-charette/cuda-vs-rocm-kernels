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

    } // namespace pgkl
