#include "pgkl/reduction.h"

namespace pgkl {
double reduction_sum_reference(const float* data, std::size_t n) {
    double acc = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        acc += static_cast<double>(data[i]);
    }
    return acc
}

double reduction_sum_reference(const std::vector<float>& data) {
    return reduction_sum_reference(data.data(), data.size());
}

} // namespace pgkl
