#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace pgkl::hip_detail {

inline void hip_check(const hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string{operation} + ": " + hipGetErrorString(status));
    }
}

[[nodiscard]] constexpr auto ceil_div(const std::size_t value, const std::size_t divisor) noexcept -> std::size_t {
    return (value + divisor - 1U) / divisor;
}

}  // namespace pgkl::hip_detail
