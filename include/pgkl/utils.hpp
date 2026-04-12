#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

namespace pgkl {

[[nodiscard]] inline auto make_patterned_vector(const std::size_t n) -> std::vector<float> {
    auto values = std::vector<float>(n);
    for (const auto index : std::views::iota(std::size_t{0}, n)) {
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
    for (const auto row : std::views::iota(std::size_t{0}, rows)) {
        for (const auto col : std::views::iota(std::size_t{0}, cols)) {
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
    for (const auto index : std::views::iota(std::size_t{0}, n)) {
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

[[nodiscard]] inline auto vectors_nearly_equal(const std::span<const float> lhs,
                                               const std::span<const float> rhs,
                                               const float atol = 1.0e-5F,
                                               const float rtol = 1.0e-5F,
                                               std::size_t* bad_index = nullptr) -> bool {
    if (lhs.size() != rhs.size()) {
        if (bad_index != nullptr) {
            *bad_index = std::numeric_limits<std::size_t>::max();
        }
        return false;
    }

    const auto mismatch = std::ranges::mismatch(lhs, rhs, [=](const float left, const float right) {
        return nearly_equal(left, right, atol, rtol);
    });

    if (mismatch.in1 != lhs.end()) {
        if (bad_index != nullptr) {
            *bad_index = static_cast<std::size_t>(std::distance(lhs.begin(), mismatch.in1));
        }
        return false;
    }

    return true;
}

[[nodiscard]] inline auto checksum(const std::span<const float> values) -> float {
    return std::transform_reduce(values.begin(), values.end(), 0.0F, std::plus<>{}, [](const float value) {
        return value;
    });
}

}  // namespace pgkl
