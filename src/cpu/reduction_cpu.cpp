#include "pgkl/reduction.hpp"

namespace pgkl {

auto reduction_cpu(const std::span<const float> input) -> float {
    float sum = 0.0F;
    for (const auto value : input) {
        sum += value;
    }
    return sum;
}

} // namespace pgkl
