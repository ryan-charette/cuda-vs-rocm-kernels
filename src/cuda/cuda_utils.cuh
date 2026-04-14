#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace pgkl::cuda_detail {

inline void cuda_check(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string{operation} + ": " + cudaGetErrorString(status));
    }
}

[[nodiscard]] constexpr auto ceil_div(const std::size_t value, const std::size_t divisor) noexcept -> std::size_t {
    return (value + divisor - 1U) / divisor;
}

}  // namespace pgkl::cuda_detail
