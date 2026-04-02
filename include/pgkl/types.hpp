#pragma once

#include <string>

namespace pgkl {
  enum class Backend {
    CPU,
    CUDA,
    HIP
  };

  enum class Kernel {
    Reduction,
    Stencil2D,
    MatMulTiled
  };

  inline std::string to_string(Backend b) {
    switch (b) {
      case Backend::CPU   return "cpu";
      case Backend::CUDA: return "cuda";
      case Backend::HIP:  return "hip";
    }
    return "unknown";
  }

  inline std::string to_string(Kernel k) {
    switch(k) {
      case Kernel::Reduction:   return "reduction";
      case Kernel::Stencil2D:   return "stencil2d";
      case Kernel::MatMulTiled: return "matmul";
    }
    return "unknown";
  }

} // namespace pgkl
