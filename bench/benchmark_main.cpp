#include "pgkl/cli.hpp"
#include "pgkl/matmul_tiled.hpp"
#include "pgkl/reduction.hpp"
#include "pgkl/stencil2d.hpp"
#include "pgkl/utils.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct BenchResult {
    std::string metric_name;
    double metric_value{};
    double average_time_ms{};
};

template <typename Fn>
auto measure_average_ms(const int repeats, Fn&& fn) -> double {
    auto total = std::chrono::duration<double, std::milli>::zero();
    for (int iteration = 0; iteration < repeats; ++iteration) {
        const auto start = clock_type::now();
        fn();
        const auto stop = clock_type::now();
        total += stop - start;
    }
    return total.count() / static_cast<double>(repeats);
}

void print_result(const pgkl::BenchConfig& config, const BenchResult& result) {
    std::cout << std::fixed << std::setprecision(6);

    if (config.format == pgkl::OutputFormat::CSV) {
        std::cout << "backend,kernel,size,repeats,tile_size,metric_name,metric_value,avg_time_ms\n";
        std::cout << pgkl::to_string(config.backend) << ','
                  << pgkl::to_string(config.kernel) << ','
                  << config.size << ','
                  << config.repeats << ','
                  << config.tile_size << ','
                  << result.metric_name << ','
                  << result.metric_value << ','
                  << result.average_time_ms << '\n';
        return;
    }
    
    std::cout << "backend=" << pgkl::to_string(config.backend) << '\n';
    std::cout << "kernel=" << pgkl::to_string(config.kernel) << '\n';
    std::cout << "size=" << config.size << '\n';
    std::cout << "repeats=" << config.repeats << '\n';
    std::cout << "tile_size=" << config.tile_size << '\n';
    std::cout << result.metric_name << '=' << result.metric_value << '\n';
    std::cout << "avg_time_ms=" << result.average_time_ms << '\n';
        }
    }
    
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto config = pgkl::parse_args(argc, argv);

        if (config.backend == pgkl::Backend::CUDA) {
#ifndef PGKL_HAS_CUDA
            throw std::runtime_error("CUDA not supported");
#endif
        }

        BenchResult result{};
        switch (config.kernel) {
            case pgkl::Kernel::Reduction:
                result = run_reduction(config);
                break;
            case pgkl::Kernel::Stencil2D:
                result = run_stencil(config);
                break;
            case pgkl::Kernel::MatMulTiled:
                result = run_matmul(config);
                break;
        }

        print_result(config, result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark failed: " << error.what() << '\n';
        std::cerr << "usage: ./pgkl_bench --backend cpu --kernel reduction --size 1048576 [--repeats 5] [--tile-size 32] [--format text]\n";
        return 1;
    }
}
                    
