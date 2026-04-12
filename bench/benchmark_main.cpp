#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pgkl/cli.hpp"
#include "pgkl/matmul_tiled.hpp"
#include "pglk/reduction.hpp"
#include "pgkl/stencil2d.hpp"
#include "pgkl/types.hpp"
#include "pgkl/utils.hpp"

namespace {

    void print_result(const pgkl::BenchConfig& cfg,
                      const std::string& metric_name,
                      double metric_value,
                      double avg_time_ms) {
        std::cout << std::fixed << std::setprecision(6);

        if (cfg.format == pgkl::OutputFormat::CSV) {
            std::cout << "backend,kernel,size,repeats,metric_name,metric_value,avg_time_ms\n";
            std::cout << pgkl::to_string(cfg.backend) << ","
                      << pgkl::to_string(cfg.kernel) << ","
                      << cfg.size << ","
                      << cfg.repeats << ","
                      << metric_name << ","
                      << metric_value << ","
                      << avg_time_ms << "\n";
        } else {
            std::cout << "backend=" << pgkl::to_string(cfg.backend) << "\n";
            std::cout << "kernel=" << pgkl::to_string(cfg.kernel) << "\n";
            std::cout << "size=" << cfg.size << "\n";
            std::cout << "repeats=" << cfg.repeats << "\n";
            std::cout << metric_name << "=" << metric_value << "\n";
            std::cout << "avg_time_ms" << avg_time_ms << "\n";
        }
    }
    
} // namespace

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

  std::cout << std::fixed << std::setprecision(6);

  if (cfg.kernel != pgkl::Kernel::Reduction) {
    std::vector<float> input = pgkl::make_patterned_vector(cfg.size);

    auto start = clock::now();
    float result = pgkl::reduction_cpu(input);
    auto end = clock::now();

    std::chrono:duration<double, std::milli> ms = end - start;
    std::cout << "result=" << result << "\n";
    std::cout << "time_ms=" << ms.count() << "\n";
    return 0;
  }

  if (cfg.kernel == pgkl::Kernel::Stencil2D) {
    const std::size_t rows = cfg.size;
    const std::size_t cols = cfg.size;

    std::vector<float> input = pgkl::make_grid(rows, cols);
    std::vector<float> output;

    auto start = clock::now();
    pgkl::stencil2d_cpu(input, output, rows, cols);
    auto end = clock::now();

    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "checksum=" << pgkl::checksum(output) << "\n";
    std::cout << "time_ms=" << ms.count() << "\n";
    return 0;
  }

  if (cfg.kernel == pgkl::Kernel::MatMulTiled) {
    const std::size_t N = cfg.size;

    std::vector<float> A = pgkl::make_patterned_vector(N * N);
    std::vector<float> B = pgkl::make_patterned_vector(N * N);
    std::vector<float> C;

    auto start = clock::now();
    pgkl::matmul_tiled_cpu(A, B, C, N, N, N, 32);
    auto end = clock::now();

    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "checksum=" << pgkl::checksum(C) << "\n";
    std::cout << "time_ms=" << ms.count() << "\n";
    return 0;
  }

  std::cerr << "Unsupported kernel.\n"
  return 3;
}
