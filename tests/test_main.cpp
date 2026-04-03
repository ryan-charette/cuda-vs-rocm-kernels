#include <cmath>
#include <iostream>
#include <vector>

#include "pgkl/reduction.hpp"

int main() {
  std::vector<float> input = {1.f, 2.f, 3.f, 4.f, 5.f};
  float got = pgkl::reduction_cpu(input);
  float expected = 15.0f;

  if (std::fabs(got - expected) > 1e-6f) {
    std::cerr << "FAIL: expected " << expected << ", got " << got << "\n";
    return 1;
  }

  std::cout << "PASS: reduction_cpu basic correctness\n";
  return 0;
}
