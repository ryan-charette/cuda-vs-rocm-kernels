#pragma once

#include "pgkl/timing.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace pgkl::cuda_detail {

inline void cuda_check(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string{operation} + ": " + cudaGetErrorString(status));
    }
}

[[nodiscard]] constexpr auto ceil_div(const std::size_t value, const std::size_t divisor) noexcept -> std::size_t {
    return (value + divisor - 1U) / divisor;
}

template <typename LaunchFn>
void launch_timed_kernel(LaunchFn&& launch,
                         TimingResult* timing,
                         const char* launch_operation,
                         const char* sync_operation) {
    if (timing == nullptr) {
        std::forward<LaunchFn>(launch)();
        cuda_check(cudaGetLastError(), launch_operation);
        cuda_check(cudaDeviceSynchronize(), sync_operation);
        return;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    try {
        cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
        cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
        cuda_check(cudaEventRecord(start, 0), "cudaEventRecord start");

        std::forward<LaunchFn>(launch)();

        cuda_check(cudaEventRecord(stop, 0), "cudaEventRecord stop");
        cuda_check(cudaGetLastError(), launch_operation);
        cuda_check(cudaEventSynchronize(stop), sync_operation);

        float elapsed_ms = 0.0F;
        cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
        timing->add_kernel_time_ms(static_cast<double>(elapsed_ms));
    } catch (...) {
        if (stop != nullptr) {
            static_cast<void>(cudaEventDestroy(stop));
        }
        if (start != nullptr) {
            static_cast<void>(cudaEventDestroy(start));
        }
        throw;
    }

    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy stop");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy start");
}

}  // namespace pgkl::cuda_detail
