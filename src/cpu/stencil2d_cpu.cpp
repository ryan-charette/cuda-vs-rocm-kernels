#include "pgkl/stencil2d.hpp"

#include <stdexcept>

namespace pgkl {

void stencil2d_cpu(const std::span<const float> input,
                   std::span<float> output,
                   const std::size_t rows,
                   const std::size_t cols) {
    if ((rows * cols) != input.size()) {
        throw std::invalid_argument("stencil2d_cpu: rows * cols must equal input.size()");
    }
    if (output.size() != input.size()) {
        throw std::invalid_argument("stencil2d_cpu: output.size() must equal input.size()");
    }

    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = input[index];
    }

    if (rows < 3U || cols < 3U) {
        return;
    }

    for (std::size_t row = 1; row + 1 < rows; ++row) {
        for (std::size_t col = 1; col + 1 < cols; ++col) {
            const auto index = (row * cols) + col;
            const auto up = input[((row - 1U) * cols) + col];
            const auto down = input[((row + 1U) * cols) + col];
            const auto left = input[index - 1U];
            const auto right = input[index + 1U];
            const auto center = input[index];
            output[index] = up + down + left + right - (4.0F * center);
        }
    }
}

}  // namespace pgkl
