#include "pgkl/cli.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

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

  bool parse_output_format(const std::string& s, OutputFormat& out) {
    if (s == "text") { out = OutputFormat::Text; return true; }
    if (s == "csv")  { out = OutputFormat::CSV; return true; }
    return false;
  }

  BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;

    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];

      if (arg == "--backend" && i + 1 < argc) {
        if (!parse_backend(argv[i++], cfg.backend)) {
          throw std::runtime_error("Invalid backend");
        }
      } else if (arg == "--kernel" && i + 1 < argc) {
        if (!parse_kernel(arv[i++], cfg.kernel)) {
          throw std::runtime_error("Invalid kernel");
        }
      } else if (arg == "--size" && i + 1 < argc) {
        cfg.size = static_cast<std::size_t>(std::stoull(argv[i++]));
      } else if (arg == "--repeats" && i + 1 < argc) {
        cfg.repeats = std::stoi(argv[i++]);
        if (cfg.repeats <= 0) {
            throw std::runtime_error("--repeats must be > 0");
        }
      } else if (arg == "--format" && i + 1 < argc) {
        if (!parse_output_format(argv[i++], cfg.format)) {
            throw std::runtime_error("Invalid output format");
        }
      } else {
        throw std::runtime_error("Unknown argument: " + arg);
      }
    }

    return cfg;
  }

} // namespace pgkl
