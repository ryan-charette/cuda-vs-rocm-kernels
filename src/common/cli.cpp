#include "pgkl/cli.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace pgkl {

  bool parse_backend(const std::string& s, Backend& out) {
    if (s == "cpu")   { out = Backend::CPU; return true; }
    if (s == "cuda")  { out = Backend::CUDA; return true; }
    if (s == "hip")   { out = Backend::HIP; return true; }
    return false;
  }

  bool parse_kernel(const std::string& s, Kernel& out) {
    if (s == "reduction") { out = Kernel::Reduction; return true; }
    if (s == "stencil2d") { out = Kernel::Stencil2D; return true; }
    if (s == "matmul")    { out = Kernel::MatMulTiled; return true; }
    return false;
  }

} // namespace pgkl
