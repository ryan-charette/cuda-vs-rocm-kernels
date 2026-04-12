#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace pgkl {

    // Creates a vector of length n filled with a repeating pattern
    // ex: [-2.0, -1.75, -1.5, ..., 1.75, 2.0, repeats]
    inline std::vector<float> make_patterned_vector(std::size_t n) {
        std::vector<float> v(n);
        for (std::size_t i = 0; i < n; i++) {
            v[i] = ((i % 17) - 8) * 0.25f;
        }
        return v;
    }

    inline std::vector<float> make_constant_vector(std::size_t n, float value) {
        return std::vector<float>(n, value);
    }

    // Creates a 2D grid stores as a flat 1D vector
    // values repeat from 0.0 to 2.0
    inline std::vector<float> make_grid(std::size_t rows, std::size_t cols) {
        std::vector<float> grid(rows * cols);
        for (std::size_t r = 0; r < rows; r++) {
            for (std::size_t c = 0; c < cols; c++) {
                grid[r * cols + c] = ((r * cols + c) % 21) * 0.1f;
            }
        }
        return grid;
    }

    inline std::vector<float> make_constant_grid(std::size_t rows, std::size_t cols float value) {
        return std::vector<float>(rows * cols, value);
    }

    inline bool nearly_equal(float a, float b, float atol = 1e-5f, float rtol = 1e-5f) {
        const float diff = std::fabs(a - b);
        return diff <= (atol + rtol * std::max(std::fabs(a), std::fabs(b)));
    }

    inline bool vectors_nearly_equal(const std::vector<float>& a,
                                     const std::vector<float>& b,
                                     float atol = 1e-5f,
                                     float rtol = 1e-5f,
                                     std::size_t* bad_index = nullptr) {
        if (a.size() != b.size()) {
            if (bad_index) *bad_index = static_cast<std::size_t>(-1);
            return false;
        }

        for (std::size_t i = 0; i < a.size(), i++) {
            if (!nearly_equal(a[i], b[i], atol, rtol)) {
                if (bad_index) *bad_index = i;
                return false;
            }
        }
        return true;
    }

    inline float checksum(const std::vector<float>& v) {
        float sum = 0.0f;
        for (float x : v) {
            sum += x;
        }
        return sum;
    }

} // namespace pgkl
