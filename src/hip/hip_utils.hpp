#pragma once

#include "pgkl/timing.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace pgkl::hip_detail {

inline void hip_check(const hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string{operation} + ": " + hipGetErrorString(status));
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
        hip_check(hipGetLastError(), launch_operation);
        hip_check(hipDeviceSynchronize(), sync_operation);
        return;
    }

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;

    try {
        hip_check(hipEventCreate(&start), "hipEventCreate start");
        hip_check(hipEventCreate(&stop), "hipEventCreate stop");
        hip_check(hipEventRecord(start, 0), "hipEventRecord start");

        std::forward<LaunchFn>(launch)();

        hip_check(hipEventRecord(stop, 0), "hipEventRecord stop");
        hip_check(hipGetLastError(), launch_operation);
        hip_check(hipEventSynchronize(stop), sync_operation);

        float elapsed_ms = 0.0F;
        hip_check(hipEventElapsedTime(&elapsed_ms, start, stop), "hipEventElapsedTime");
        timing->add_kernel_time_ms(static_cast<double>(elapsed_ms));
    } catch (...) {
        if (stop != nullptr) {
            static_cast<void>(hipEventDestroy(stop));
        }
        if (start != nullptr) {
            static_cast<void>(hipEventDestroy(start));
        }
        throw;
    }

    hip_check(hipEventDestroy(stop), "hipEventDestroy stop");
    hip_check(hipEventDestroy(start), "hipEventDestroy start");
}

}  // namespace pgkl::hip_detail
