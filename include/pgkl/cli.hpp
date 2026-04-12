#pragma once

#include <cstddef>
#include <string>

#include "pgkl/types.hpp"

namespace pgkl {

    enum class OutputFormat {
        Text,
        CSV
    };

    inline std::string_to_string(OutputFormat f) {
        switch(f) {
            case OutputFormat::Text: return "text";
            case OutputFormat::CSV:  return "csv";
        }
        return "unknown";
    }

    struct BenchConfig {
        Backend backend = Backend::CPU;
        Kernel kernel = Kernel::Reduction;
        std::size_t size = 1 << 20;
        int repeats = 5;
        OutputFormat format = OutputFormat::Text;
    };

    bool parse_backend(const std::string& s, Backend& out);
    bool parse_kernel(const std::string& s, Kernel& out);
    bool parse_output_format(const std::string& s, OutputFormat& out);
    BenchConfig parse_args(int argc, char** argv);

} // namespace pgkl
