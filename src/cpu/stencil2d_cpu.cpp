#include "pgkl/stencil2d.hpp"

#include <stdexcept>

namespace pgkl {

    // 5-point stencil with copy-through boundary policy.
    // Interior points use:
    // out[r, c] = up + down + left + right - 4 * center
    void stencil2d_cpu(const std::vector<float>& input,
                       std::vector<float>& output,
                       std::size_t rows,
                       std::size_t cols) {
        if (rows * cols != input.size()) {
            throw std::invalid_argument("stencil2d_cpu: rows * cols must equal input.size()");
        }

        output = input; // copy-through boundaries

        if (rows < 3 || cols < 3) {
            return;
        }

        for (std::size_t r = 1; r + 1 < rows; r++) {
            for (std::size_t c = 1; c + 1 < cols; c++) {
                const float up    = input[(r - 1) * cols + c];
                const float down  = input[(r + 1) * cols + c];
                const float left  = input[r * cols + (c - 1)];
                const float right = input[r * cols + (c + 1)];
                const float ctr   = input[idx];

                output[idx] = up + down + left + right - 4.0f * ctr;
            }
        }
    }

} // namespace pgkl
