#include "pgkl/metadata.hpp"

#include "cuda_utils.cuh"

#include <cuda_runtime.h>

#include <string>

namespace pgkl {

auto metadata_cuda() -> RuntimeMetadata {
    auto metadata = RuntimeMetadata{};
    metadata.device_vendor = "NVIDIA";
    metadata.compiler = compiler_description();
    metadata.cxx_standard = cxx_standard_description();

    int device = 0;
    cuda_detail::cuda_check(cudaGetDevice(&device), "cudaGetDevice");

    cudaDeviceProp properties{};
    cuda_detail::cuda_check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    metadata.device_name = properties.name;

    int runtime_version = 0;
    int driver_version = 0;
    cuda_detail::cuda_check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    cuda_detail::cuda_check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    metadata.runtime_version = "cuda_runtime=" + std::to_string(runtime_version);
    metadata.driver_version = "cuda_driver=" + std::to_string(driver_version);

    return metadata;
}

}  // namespace pgkl
