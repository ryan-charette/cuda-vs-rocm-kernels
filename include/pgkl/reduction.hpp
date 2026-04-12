#pragma once

#include <vector>

namespace pgkl {

    float reduction_cpu(const std::vector<float>& input);
    float reduction_cuda(const std::vector<float>& input);

} // namespace pgkl
