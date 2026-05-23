#include "pgkl/cli.hpp"
#include "pgkl/matmul_tiled.hpp"
#include "pgkl/metadata.hpp"
#include "pgkl/reduction.hpp"
#include "pgkl/stencil2d.hpp"
#include "pgkl/timing.hpp"
#include "pgkl/utils.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct Measurement {
    double average_end_to_end_ms{};
    double average_kernel_ms{};
    bool kernel_timing_available{false};
    bool correct{true};
    int checked_runs{};
};

struct BenchResult {
    std::string metric_name;
    double metric_value{};
    Measurement measurement;
    pgkl::RuntimeMetadata metadata;
};

[[nodiscard]] auto bool_text(const bool value) -> const char* {
    return value ? "true" : "false";
}

[[nodiscard]] auto csv_escape(const std::string_view value) -> std::string {
    const auto needs_quotes = value.find_first_of(",\"\n\r") != std::string_view::npos;
    if (!needs_quotes) {
        return std::string{value};
    }

    auto escaped = std::string{"\""};
    for (const char character : value) {
        if (character == '"') {
            escaped += "\"\"";
        } else {
            escaped += character;
        }
    }
    escaped += '"';
    return escaped;
}

[[nodiscard]] auto unsupported_backend_message(const pgkl::Backend backend) -> std::string {
    return std::string{pgkl::to_string(backend)} + " not supported by this build";
}

void ensure_backend_supported(const pgkl::Backend backend) {
    if (backend == pgkl::Backend::CUDA) {
#ifndef PGKL_HAS_CUDA
        throw std::runtime_error("CUDA not supported by this build");
#endif
    }

    if (backend == pgkl::Backend::HIP) {
#ifndef PGKL_HAS_HIP
        throw std::runtime_error("HIP not supported by this build");
#endif
    }

    if (backend == pgkl::Backend::SYCL) {
#ifndef PGKL_HAS_SYCL
        throw std::runtime_error("SYCL not supported by this build");
#endif
    }
}

[[nodiscard]] auto collect_metadata(const pgkl::Backend backend) -> pgkl::RuntimeMetadata {
    switch (backend) {
        case pgkl::Backend::CPU:
            return pgkl::metadata_cpu();
        case pgkl::Backend::CUDA:
#ifdef PGKL_HAS_CUDA
            return pgkl::metadata_cuda();
#else
            break;
#endif
        case pgkl::Backend::HIP:
#ifdef PGKL_HAS_HIP
            return pgkl::metadata_hip();
#else
            break;
#endif
        case pgkl::Backend::SYCL:
#ifdef PGKL_HAS_SYCL
            return pgkl::metadata_sycl();
#else
            break;
#endif
    }

    throw std::runtime_error(unsupported_backend_message(backend));
}

template <typename RunFn, typename ValidateFn>
auto measure_runs(const pgkl::BenchConfig& config, RunFn&& run, ValidateFn&& validate) -> Measurement {
    for (int warmup = 0; warmup < config.warmups; ++warmup) {
        run(nullptr);
    }

    auto measurement = Measurement{};
    auto total_end_to_end = std::chrono::duration<double, std::milli>::zero();
    double total_kernel_ms = 0.0;
    int kernel_samples = 0;

    for (int iteration = 0; iteration < config.repeats; ++iteration) {
        auto timing = pgkl::TimingResult{};

        const auto start = clock_type::now();
        run(&timing);
        const auto stop = clock_type::now();

        const auto elapsed = stop - start;
        total_end_to_end += elapsed;

        if (timing.kernel_time_available) {
            total_kernel_ms += timing.kernel_time_ms;
            ++kernel_samples;
        } else if (config.backend == pgkl::Backend::CPU) {
            total_kernel_ms += std::chrono::duration<double, std::milli>{elapsed}.count();
            ++kernel_samples;
        }

        if (config.check_correctness) {
            ++measurement.checked_runs;
            measurement.correct = validate() && measurement.correct;
        }
    }

    measurement.average_end_to_end_ms = total_end_to_end.count() / static_cast<double>(config.repeats);
    measurement.kernel_timing_available = kernel_samples == config.repeats;
    if (measurement.kernel_timing_available) {
        measurement.average_kernel_ms = total_kernel_ms / static_cast<double>(kernel_samples);
    }

    return measurement;
}

auto dispatch_reduction(const pgkl::Backend backend,
                        const std::vector<float>& input,
                        pgkl::TimingResult* timing) -> float {
    if (backend == pgkl::Backend::CPU) {
        return pgkl::reduction_cpu(input, timing);
    }
#ifdef PGKL_HAS_CUDA
    if (backend == pgkl::Backend::CUDA) {
        return pgkl::reduction_cuda(input, timing);
    }
#endif
#ifdef PGKL_HAS_HIP
    if (backend == pgkl::Backend::HIP) {
        return pgkl::reduction_hip(input, timing);
    }
#endif
#ifdef PGKL_HAS_SYCL
    if (backend == pgkl::Backend::SYCL) {
        return pgkl::reduction_sycl(input, timing);
    }
#endif

    throw std::runtime_error(unsupported_backend_message(backend));
}

void dispatch_stencil(const pgkl::Backend backend,
                      const std::vector<float>& input,
                      std::vector<float>& output,
                      const std::size_t side,
                      pgkl::TimingResult* timing) {
    if (backend == pgkl::Backend::CPU) {
        pgkl::stencil2d_cpu(input, output, side, side, timing);
        return;
    }
#ifdef PGKL_HAS_CUDA
    if (backend == pgkl::Backend::CUDA) {
        pgkl::stencil2d_cuda(input, output, side, side, timing);
        return;
    }
#endif
#ifdef PGKL_HAS_HIP
    if (backend == pgkl::Backend::HIP) {
        pgkl::stencil2d_hip(input, output, side, side, timing);
        return;
    }
#endif
#ifdef PGKL_HAS_SYCL
    if (backend == pgkl::Backend::SYCL) {
        pgkl::stencil2d_sycl(input, output, side, side, timing);
        return;
    }
#endif

    throw std::runtime_error(unsupported_backend_message(backend));
}

void dispatch_matmul(const pgkl::Backend backend,
                     const std::vector<float>& a,
                     const std::vector<float>& b,
                     std::vector<float>& c,
                     const std::size_t side,
                     const std::size_t tile_size,
                     pgkl::TimingResult* timing) {
    if (backend == pgkl::Backend::CPU) {
        pgkl::matmul_tiled_cpu(a, b, c, side, side, side, tile_size, timing);
        return;
    }
#ifdef PGKL_HAS_CUDA
    if (backend == pgkl::Backend::CUDA) {
        pgkl::matmul_tiled_cuda(a, b, c, side, side, side, tile_size, timing);
        return;
    }
#endif
#ifdef PGKL_HAS_HIP
    if (backend == pgkl::Backend::HIP) {
        pgkl::matmul_tiled_hip(a, b, c, side, side, side, tile_size, timing);
        return;
    }
#endif
#ifdef PGKL_HAS_SYCL
    if (backend == pgkl::Backend::SYCL) {
        pgkl::matmul_tiled_sycl(a, b, c, side, side, side, tile_size, timing);
        return;
    }
#endif

    throw std::runtime_error(unsupported_backend_message(backend));
}

auto run_reduction(const pgkl::BenchConfig& config) -> BenchResult {
    const auto input = pgkl::make_patterned_vector(config.size);
    const auto expected = config.check_correctness ? pgkl::reduction_cpu(input) : 0.0F;
    auto result = 0.0F;

    auto measurement = measure_runs(
        config,
        [&](pgkl::TimingResult* timing) { result = dispatch_reduction(config.backend, input, timing); },
        [&] { return pgkl::nearly_equal(result, expected, 1.0e-3F, 1.0e-3F); });

    return BenchResult{"result", static_cast<double>(result), measurement, collect_metadata(config.backend)};
}

auto run_stencil(const pgkl::BenchConfig& config) -> BenchResult {
    const auto side = config.size;
    const auto input = pgkl::make_grid(side, side);
    auto expected = std::vector<float>(input.size(), 0.0F);
    if (config.check_correctness) {
        pgkl::stencil2d_cpu(input, expected, side, side);
    }

    auto output = std::vector<float>(input.size(), 0.0F);
    auto measurement = measure_runs(
        config,
        [&](pgkl::TimingResult* timing) { dispatch_stencil(config.backend, input, output, side, timing); },
        [&] {
            return pgkl::vectors_nearly_equal(output, expected, 1.0e-3F, 1.0e-3F);
        });

    return BenchResult{"checksum", static_cast<double>(pgkl::checksum(output)), measurement, collect_metadata(config.backend)};
}

auto run_matmul(const pgkl::BenchConfig& config) -> BenchResult {
    const auto side = config.size;
    const auto a = pgkl::make_patterned_vector(side * side);
    const auto b = pgkl::make_patterned_vector(side * side);
    auto expected = std::vector<float>(side * side, 0.0F);
    if (config.check_correctness) {
        pgkl::matmul_tiled_cpu(a, b, expected, side, side, side, config.tile_size);
    }

    auto c = std::vector<float>(side * side, 0.0F);
    auto measurement = measure_runs(
        config,
        [&](pgkl::TimingResult* timing) {
            dispatch_matmul(config.backend, a, b, c, side, config.tile_size, timing);
        },
        [&] {
            return pgkl::vectors_nearly_equal(c, expected, 1.0e-3F, 1.0e-3F);
        });

    return BenchResult{"checksum", static_cast<double>(pgkl::checksum(c)), measurement, collect_metadata(config.backend)};
}

void print_result(const pgkl::BenchConfig& config, const BenchResult& result) {
    std::cout << std::fixed << std::setprecision(6);

    if (config.format == pgkl::OutputFormat::CSV) {
        std::cout
            << "backend,kernel,size,repeats,warmups,tile_size,check_correctness,checked_runs,correct,"
               "metric_name,metric_value,avg_end_to_end_ms,avg_kernel_ms,kernel_timing_available,"
               "device_name,device_vendor,runtime_version,driver_version,compiler,cxx_standard\n";
        std::cout << pgkl::to_string(config.backend) << ','
                  << pgkl::to_string(config.kernel) << ','
                  << config.size << ','
                  << config.repeats << ','
                  << config.warmups << ','
                  << config.tile_size << ','
                  << bool_text(config.check_correctness) << ','
                  << result.measurement.checked_runs << ','
                  << bool_text(result.measurement.correct) << ','
                  << csv_escape(result.metric_name) << ','
                  << result.metric_value << ','
                  << result.measurement.average_end_to_end_ms << ','
                  << result.measurement.average_kernel_ms << ','
                  << bool_text(result.measurement.kernel_timing_available) << ','
                  << csv_escape(result.metadata.device_name) << ','
                  << csv_escape(result.metadata.device_vendor) << ','
                  << csv_escape(result.metadata.runtime_version) << ','
                  << csv_escape(result.metadata.driver_version) << ','
                  << csv_escape(result.metadata.compiler) << ','
                  << csv_escape(result.metadata.cxx_standard) << '\n';
        return;
    }

    std::cout << "backend=" << pgkl::to_string(config.backend) << '\n';
    std::cout << "kernel=" << pgkl::to_string(config.kernel) << '\n';
    std::cout << "size=" << config.size << '\n';
    std::cout << "repeats=" << config.repeats << '\n';
    std::cout << "warmups=" << config.warmups << '\n';
    std::cout << "tile_size=" << config.tile_size << '\n';
    std::cout << "check_correctness=" << bool_text(config.check_correctness) << '\n';
    std::cout << "checked_runs=" << result.measurement.checked_runs << '\n';
    std::cout << "correct=" << bool_text(result.measurement.correct) << '\n';
    std::cout << result.metric_name << '=' << result.metric_value << '\n';
    std::cout << "avg_end_to_end_ms=" << result.measurement.average_end_to_end_ms << '\n';
    std::cout << "avg_kernel_ms=" << result.measurement.average_kernel_ms << '\n';
    std::cout << "kernel_timing_available=" << bool_text(result.measurement.kernel_timing_available) << '\n';
    std::cout << "device_name=" << result.metadata.device_name << '\n';
    std::cout << "device_vendor=" << result.metadata.device_vendor << '\n';
    std::cout << "runtime_version=" << result.metadata.runtime_version << '\n';
    std::cout << "driver_version=" << result.metadata.driver_version << '\n';
    std::cout << "compiler=" << result.metadata.compiler << '\n';
    std::cout << "cxx_standard=" << result.metadata.cxx_standard << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto config = pgkl::parse_args(argc, argv);
        ensure_backend_supported(config.backend);

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
        return result.measurement.correct ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "benchmark failed: " << error.what() << '\n';
        std::cerr << "usage: ./pgkl_bench --backend cpu|cuda|hip|sycl --kernel reduction|stencil2d|matmul "
                     "--size N [--repeats 5] [--warmups 1] [--tile-size 32] [--format text|csv] "
                     "[--skip-correctness]\n";
        return 1;
    }
}
