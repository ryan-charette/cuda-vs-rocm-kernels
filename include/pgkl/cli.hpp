#pragma once

#include <cstddef>
#include <optional>
#include <string_view>

#include "pgkl/types.hpp"

namespace pgkl {

enum class OutputFormat {
    Text,
    CSV,
};

[[nodiscard]] constexpr std::string_view to_string(const OutputFormat format) noexcept {
    switch (format) {
        case OutputFormat::Text:
            return "text";
        case OutputFormat::CSV:
            return "csv";
    }
    return "unknown";
}

struct BenchConfig {
    Backend backend{Backend::CPU};
    Kernel kernel{Kernel::Reduction};
    std::size_t size{1u << 20};
    int repeats{5};
    std::size_t tile_size{32};
    OutputFormat format{OutputFormat::Text};
};

[[nodiscard]] auto parse_backend(std::string_view value) -> std::optional<Backend>;
[[nodiscard]] auto parse_kernel(std::string_view value) -> std::optional<Kernel>;
[[nodiscard]] auto parse_output_format(std::string_view value) -> std::optional<OutputFormat>;
[[nodiscard]] auto parse_args(int argc, char** argv) -> BenchConfig;

}  // namespace pgkl
