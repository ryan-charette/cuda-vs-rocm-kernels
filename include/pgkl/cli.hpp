#pragma once

#include <string>
#include "pgkl/types.hpp"

namespace pgkl {

  struct BenchConfig {
    Backend backend = Backend::CPU;
    Kernel kernel = Kernel::Reduction;
    std::size_t size = 1 << 20;
  };

  bool parse_backend(const std::string& s, Backend& out);
  bool parse_kernel(const std::string& s, Kernel& out);
  BenchConfig parse_args(int argc, char** argv);

} // namespace pgkl
