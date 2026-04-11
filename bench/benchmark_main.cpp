#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "pgkl/cli.hpp"
#include "pgkl/reduction.hpp"
#include "pgkl/types.hpp"

int main(int argc, char** argv) {
  using clock = std::chrono::high_resolution_clock;

  pgkl::BenchConfig cfg;
  try {
    cfg = pgkl::parse_args(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Argument error: " << e.what() << "\n";
    std::cerr << "Usage: ./pgkl_bench --backend cpu --kernel reduction --size 1048576\n";
    return 1;
  }

  std::cout << "backend=" << pgkl::to_string(cfg.backend)
            << " kernel=" << pgkl::to_string(cfg.kernel)
            << " size=" << cfg.size << "\n";

  if (cfg.backend != pgkl::Backend::CPU) {
    std::cerr << "Not implemented.\n";
    return 2;
  }

  if (cfg.kernel != pgkl::Kernel::Reduction) {
    std::cerr << "Not implemented.\n";
    return 3;
  }

  std::vector<float> input(cfg.size, 1.0f);

  auto start = clock::now();
  float result = pgkl::reduction_cpu(input);
  auto end = clock::now();

  std::chrono::duration<double, std::milli> ms = end - start;

  std::cout << std::fixed << std:setprecision(3);
  std::cout << "result=" << result << "\n";
  std::cout << "time_ms=" << ms.count() << "\n";

  return 0;
}
