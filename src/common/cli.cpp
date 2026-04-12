#include "pgkl/cli.hpp"

#include <charconv>
#include <stdexcept>
#include <string>

namespace {

template <typename Integer>
auto parse_integer(const char* text, const char* option_name) -> Integer {
    Integer value{};
    const auto* begin = text;
    const auto* end = begin + std::char_traits<char>::length(text);
    const auto [ptr, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || ptr != end) {
        throw std::runtime_error(std::string{"invalid value for "} + option_name + ": " + text);
    }
    return value;
}

}  // namespace

namespace pgkl {

auto parse_backend(const std::string_view value) -> std::optional<Backend> {
    if (value == "cpu") {
        return Backend::CPU;
    }
    if (value == "cuda") {
        return Backend::CUDA;
    }
    if (value == "hip") {
        return Backend::HIP;
    }
    return std::nullopt;
}

auto parse_kernel(const std::string_view value) -> std::optional<Kernel> {
    if (value == "reduction") {
        return Kernel::Reduction;
    }
    if (value == "stencil2d") {
        return Kernel::Stencil2D;
    }
    if (value == "matmul") {
        return Kernel::MatMulTiled;
    }
    return std::nullopt;
}

auto parse_output_format(const std::string_view value) -> std::optional<OutputFormat> {
    if (value == "text") {
        return OutputFormat::Text;
    }
    if (value == "csv") {
        return OutputFormat::CSV;
    }
    return std::nullopt;
}

auto parse_args(const int argc, char** argv) -> BenchConfig {
    auto config = BenchConfig{};

    for (int index = 1; index < argc; ++index) {
        const auto argument = std::string_view{argv[index]};

        auto require_value = [&](const char* option_name) -> const char* {
            if ((index + 1) >= argc) {
                throw std::runtime_error(std::string{"missing value for "} + option_name);
            }
            ++index;
            return argv[index];
        };

        if (argument == "--backend") {
            const auto parsed = parse_backend(require_value("--backend"));
            if (!parsed.has_value()) {
                throw std::runtime_error("invalid backend");
            }
            config.backend = *parsed;
            continue;
        }

        if (argument == "--kernel") {
            const auto parsed = parse_kernel(require_value("--kernel"));
            if (!parsed.has_value()) {
                throw std::runtime_error("invalid kernel");
            }
            config.kernel = *parsed;
            continue;
        }

        if (argument == "--size") {
            config.size = parse_integer<std::size_t>(require_value("--size"), "--size");
            continue;
        }

        if (argument == "--repeats") {
            config.repeats = parse_integer<int>(require_value("--repeats"), "--repeats");
            if (config.repeats <= 0) {
                throw std::runtime_error("--repeats must be greater than zero");
            }
            continue;
        }

        if (argument == "--tile-size") {
            config.tile_size = parse_integer<std::size_t>(require_value("--tile-size"), "--tile-size");
            if (config.tile_size == 0U) {
                throw std::runtime_error("--tile-size must be greater than zero");
            }
            continue;
        }

        if (argument == "--format") {
            const auto parsed = parse_output_format(require_value("--format"));
            if (!parsed.has_value()) {
                throw std::runtime_error("invalid output format");
            }
            config.format = *parsed;
            continue;
        }

        throw std::runtime_error(std::string{"unknown argument: "} + std::string{argument});
    }

    return config;
}

}  // namespace pgkl
