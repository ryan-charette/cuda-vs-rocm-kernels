#pragma once

#include <cstddef>
#include <vector>

namespace pgkl {

    double reduction_sum_reference(const float* data, std::size_t n);
    double reduction_sum_reference(const std::vector<float>& data);

} // namespace pgkl
