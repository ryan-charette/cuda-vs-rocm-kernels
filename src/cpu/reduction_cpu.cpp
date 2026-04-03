#include "pgkl/reduction.hpp"

namespace pgkl {

  float reduction_cpu(const std::vector<float>& input) {
    float sum = 0.0f;
    for (float x : input) {
      sum += x;
    }
    return sum;
  }

} // namespace pgkl
