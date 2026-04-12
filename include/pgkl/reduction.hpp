#pragma once

#include <span>

namespace pgkl {

[[nodiscard]] auto reduction_cpu(std::span<const float> input) -> float;
[[nodiscard]] auto reduction_cuda(std::span<const float> input) -> float;

}  // namespace pgkl
