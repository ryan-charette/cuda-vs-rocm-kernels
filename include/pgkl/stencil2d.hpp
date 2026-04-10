#pragma once

#include <cstddef>
#include <vector>

namespace pgkl {

    void stencil2d_cpu(const std::vector<float>& input,
                       std::vecot<float>& output,
                       std::size_t rows,
                       std::size_t cols);

} // namespace pgkl
