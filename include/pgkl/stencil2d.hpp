#pragma once

#include <cstddef>
#include <span>

namespace pgkl {

void stencil2d_cpu(std::span<const float> input,
                   std::span<float> output,
                   std::size_t rows,
                   std::size_t cols);

void stencil2d_cuda(std::span<const float> input,
                    std::span<float> output,
                    std::size_t rows,
                    std::size_t cols);

void stencil2d_hip(std::span<const float> input,
                   std::span<float> output,
                   std::size_t rows,
                   std::size_t cols);

void stencil2d_sycl(std::span<const float> input,
                    std::span<float> output,
                    std::size_t rows,
                    std::size_t cols);

}  // namespace pgkl
