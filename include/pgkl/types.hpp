#pragma once

#include <ostream>
#include <string_view>

namespace pgkl {

enum class Backend {
    CPU,
    CUDA,
    HIP,
};

enum class Kernel {
    Reduction,
    Stencil2D,
    MatMulTiled,
};

[[nodiscard]] constexpr std::string_view to_string(const Backend backend) noexcept {
    switch (backend) {
        case Backend::CPU:
            return "cpu";
        case Backend::CUDA:
            return "cuda";
        case Backend::HIP:
            return "hip";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(const Kernel kernel) noexcept {
    switch (kernel) {
        case Kernel::Reduction:
            return "reduction";
        case Kernel::Stencil2D:
            return "stencil2d";
        case Kernel::MatMulTiled:
            return "matmul";
    }
    return "unknown";
}

inline auto operator<<(std::ostream& os, const Backend backend) -> std::ostream& {
    return os << to_string(backend);
}

inline auto operator<<(std::ostream& os, const Kernel kernel) -> std::ostream& {
    return os << to_string(kernel);
}

}  // namespace pgkl
