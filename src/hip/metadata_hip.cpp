#include "pgkl/metadata.hpp"

#include "hip_utils.hpp"

#include <hip/hip_runtime.h>

#include <string>

namespace pgkl {

auto metadata_hip() -> RuntimeMetadata {
    auto metadata = RuntimeMetadata{};
    metadata.device_vendor = "AMD";
    metadata.compiler = compiler_description();
    metadata.cxx_standard = cxx_standard_description();

    int device = 0;
    hip_detail::hip_check(hipGetDevice(&device), "hipGetDevice");

    hipDeviceProp_t properties{};
    hip_detail::hip_check(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
    metadata.device_name = properties.name;

    int runtime_version = 0;
    int driver_version = 0;
    hip_detail::hip_check(hipRuntimeGetVersion(&runtime_version), "hipRuntimeGetVersion");
    hip_detail::hip_check(hipDriverGetVersion(&driver_version), "hipDriverGetVersion");
    metadata.runtime_version = "hip_runtime=" + std::to_string(runtime_version);
    metadata.driver_version = "hip_driver=" + std::to_string(driver_version);

    return metadata;
}

}  // namespace pgkl
