#pragma once

#include <string>

namespace pgkl {

struct RuntimeMetadata {
    std::string device_name{"unknown"};
    std::string device_vendor{"unknown"};
    std::string runtime_version{"unknown"};
    std::string driver_version{"unknown"};
    std::string compiler{"unknown"};
    std::string cxx_standard{"unknown"};
};

[[nodiscard]] inline auto cxx_standard_description() -> std::string {
#if defined(_MSVC_LANG)
    return std::to_string(_MSVC_LANG);
#else
    return std::to_string(__cplusplus);
#endif
}

[[nodiscard]] inline auto compiler_description() -> std::string {
#if defined(__CUDACC__)
    return std::string{"nvcc "} + std::to_string(__CUDACC_VER_MAJOR__) + "." +
           std::to_string(__CUDACC_VER_MINOR__) + "." + std::to_string(__CUDACC_VER_BUILD__);
#elif defined(__HIPCC__)
    return "hipcc";
#elif defined(__clang__)
    return std::string{"clang "} + __clang_version__;
#elif defined(_MSC_VER)
    return std::string{"msvc "} + std::to_string(_MSC_VER);
#elif defined(__GNUC__)
    return std::string{"gcc "} + __VERSION__;
#else
    return "unknown";
#endif
}

[[nodiscard]] RuntimeMetadata metadata_cpu();
[[nodiscard]] RuntimeMetadata metadata_cuda();
[[nodiscard]] RuntimeMetadata metadata_hip();
[[nodiscard]] RuntimeMetadata metadata_sycl();

}  // namespace pgkl
