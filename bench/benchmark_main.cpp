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

auto run_reduction(const pgkl::BenchConfig& config) -> BenchResult {
    const auto input = pgkl::make_patterned_vector(config.size);
    auto result = 0.0F;

    const auto average_time_ms = measure_average_ms(config.repeats, [&] {
        if (config.backend == pgkl::Backend::CPU) {
            result = pgkl::reduction_cpu(input);
            return;
        }
#ifdef PGKL_HAS_CUDA
        if (config.backend == pgkl::Backend::CUDA) {
            result = pgkl::reduction_cuda(input);
            return;
        }
#endif
#ifdef PGKL_HAS_HIP
        if (config.backend == pgkl::Backend::HIP) {
            result = pgkl::reduction_hip(input);
            return;
        }
#endif
        throw std::runtime_error(config.backend == pgkl::Backend::HIP
                                     ? "HIP not supported"
                                     : "CUDA not supported");
    });

    return BenchResult{"result", static_cast<double>(result), average_time_ms};
}

auto run_stencil(const pgkl::BenchConfig& config) -> BenchResult {
    const auto side = config.size;
    const auto input = pgkl::make_grid(side, side);
    auto output = std::vector<float>(input.size(), 0.0F);

    const auto average_time_ms = measure_average_ms(config.repeats, [&] {
        if (config.backend == pgkl::Backend::CPU) {
            pgkl::stencil2d_cpu(input, output, side, side);
            return;
        }
#ifdef PGKL_HAS_CUDA
        if (config.backend == pgkl::Backend::CUDA) {
            pgkl::stencil2d_cuda(input, output, side, side);
            return;
        }
#endif
#ifdef PGKL_HAS_HIP
        if (config.backend == pgkl::Backend::HIP) {
            pgkl::stencil2d_hip(input, output, side, side);
            return;
        }
#endif
        throw std::runtime_error(config.backend == pgkl::Backend::HIP
                                     ? "HIP not supported"
                                     : "CUDA not supported");
    });

    return BenchResult{"checksum", static_cast<double>(pgkl::checksum(output)), average_time_ms};
}

auto run_matmul(const pgkl::BenchConfig& config) -> BenchResult {
    const auto side = config.size;
    const auto a = pgkl::make_patterned_vector(side * side);
    const auto b = pgkl::make_patterned_vector(side * side);
    auto c = std::vector<float>(side * side, 0.0F);

    const auto average_time_ms = measure_average_ms(config.repeats, [&] {
        if (config.backend == pgkl::Backend::CPU) {
            pgkl::matmul_tiled_cpu(a, b, c, side, side, side, config.tile_size);
            return;
        }
#ifdef PGKL_HAS_CUDA
        if (config.backend == pgkl::Backend::CUDA) {
            pgkl::matmul_tiled_cuda(a, b, c, side, side, side, config.tile_size);
            return;
        }
#endif
#ifdef PGKL_HAS_HIP
        if (config.backend == pgkl::Backend::HIP) {
            pgkl::matmul_tiled_hip(a, b, c, side, side, side, config.tile_size);
            return;
        }
#endif
        throw std::runtime_error(config.backend == pgkl::Backend::HIP
                                     ? "HIP not supported"
                                     : "CUDA not supported");
    });

    return BenchResult{"checksum", static_cast<double>(pgkl::checksum(c)), average_time_ms};
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

        if (config.backend == pgkl::Backend::HIP) {
#ifndef PGKL_HAS_HIP
            throw std::runtime_error("HIP not supported");
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
