#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pgkl {

[[nodiscard]] inline auto make_patterned_vector(const std::size_t n) -> std::vector<float> {
    auto values = std::vector<float>(n);
    for (std::size_t index = 0; index < n; ++index) {
        const auto pattern = static_cast<int>(index % 17U) - 8;
        values[index] = static_cast<float>(pattern) * 0.25F;
    }
    return values;
}

[[nodiscard]] inline auto make_constant_vector(const std::size_t n, const float value) -> std::vector<float> {
    return std::vector<float>(n, value);
}

[[nodiscard]] inline auto make_grid(const std::size_t rows, const std::size_t cols) -> std::vector<float> {
    auto grid = std::vector<float>(rows * cols);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            grid[(row * cols) + col] = static_cast<float>(((row * cols) + col) % 21U) * 0.1F;
        }
    }
    return grid;
}

[[nodiscard]] inline auto make_constant_grid(const std::size_t rows,
                                             const std::size_t cols,
                                             const float value) -> std::vector<float> {
    return std::vector<float>(rows * cols, value);
}

[[nodiscard]] inline auto make_identity_matrix(const std::size_t n) -> std::vector<float> {
    auto matrix = std::vector<float>(n * n, 0.0F);
    for (std::size_t index = 0; index < n; ++index) {
        matrix[(index * n) + index] = 1.0F;
    }
    return matrix;
}

[[nodiscard]] inline auto nearly_equal(const float a,
                                       const float b,
                                       const float atol = 1.0e-5F,
                                       const float rtol = 1.0e-5F) -> bool {
    const auto diff = std::fabs(a - b);
    return diff <= (atol + (rtol * std::max(std::fabs(a), std::fabs(b))));
}

[[nodiscard]] inline auto vectors_nearly_equal(const std::vector<float>& lhs,
                                               const std::vector<float>& rhs,
                                               const float atol = 1.0e-5F,
                                               const float rtol = 1.0e-5F,
                                               std::size_t* bad_index = nullptr) -> bool {
    if (lhs.size() != rhs.size()) {
        if (bad_index != nullptr) {
            *bad_index = std::numeric_limits<std::size_t>::max();
        }
        return false;
    }

    for (std::size_t index = 0; index < lhs.size(); ++index) {
        if (!nearly_equal(lhs[index], rhs[index], atol, rtol)) {
            if (bad_index != nullptr) {
                *bad_index = index;
            }
            return false;
        }
    }
    return true;
}

[[nodiscard]] inline auto checksum(const std::vector<float>& values) -> float {
    float sum = 0.0F;
    for (const auto value : values) {
        sum += value;
    }
    return sum;
}

[[nodiscard]] inline auto square_dimension_from_area(const std::size_t element_count) -> std::size_t {
    const auto side = static_cast<std::size_t>(std::sqrt(static_cast<long double>(element_count)));
    if (side * side != element_count) {
        throw std::invalid_argument("size must be a perfect square for a square-grid kernel");
    }
    return side;
}

}  // namespace pgkl
