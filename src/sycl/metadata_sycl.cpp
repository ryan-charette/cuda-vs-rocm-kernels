#include "pgkl/metadata.hpp"

#include "sycl_compat.hpp"

#include <stdexcept>
#include <string>

namespace pgkl {

auto metadata_sycl() -> RuntimeMetadata {
    namespace sycl = pgkl::sycl_compat;

    try {
        auto metadata = RuntimeMetadata{};
        metadata.compiler = compiler_description();
        metadata.cxx_standard = cxx_standard_description();

        sycl::queue queue{sycl::default_selector_v};
        const auto device = queue.get_device();
        metadata.device_name = device.get_info<sycl::info::device::name>();
        metadata.device_vendor = device.get_info<sycl::info::device::vendor>();
        metadata.runtime_version = "sycl";
        metadata.driver_version = device.get_info<sycl::info::device::driver_version>();

        return metadata;
    } catch (const sycl::exception& error) {
        throw std::runtime_error(std::string{"metadata_sycl failed: "} + error.what());
    }
}

}  // namespace pgkl
