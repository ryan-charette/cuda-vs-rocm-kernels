#pragma once

#include <span>

#include "pgkl/timing.hpp"

namespace pgkl {

[[nodiscard]] auto reduction_cpu(std::span<const float> input, TimingResult* timing = nullptr) -> float;
[[nodiscard]] auto reduction_cuda(std::span<const float> input, TimingResult* timing = nullptr) -> float;
[[nodiscard]] auto reduction_hip(std::span<const float> input, TimingResult* timing = nullptr) -> float;
[[nodiscard]] auto reduction_sycl(std::span<const float> input, TimingResult* timing = nullptr) -> float;

}  // namespace pgkl
