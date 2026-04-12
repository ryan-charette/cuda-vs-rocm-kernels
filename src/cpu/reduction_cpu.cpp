#include "pgkl/reduction.hpp"

namespace pgkl {

auto reduction_cpu(const std::span<const float> input) -> float {
    float sum = 0.0F;
    for (float x : input) {
        sum += x;
    }
    return sum;
}

} // namespace pgkl
