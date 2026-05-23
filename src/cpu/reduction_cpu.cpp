#include "pgkl/reduction.hpp"

namespace pgkl {

auto reduction_cpu(const std::span<const float> input, TimingResult* timing) -> float {
    static_cast<void>(timing);

    float sum = 0.0F;
    for (const auto value : input) {
        sum += value;
    }
    return sum;
}

} // namespace pgkl
