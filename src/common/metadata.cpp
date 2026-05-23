#include "pgkl/metadata.hpp"

namespace pgkl {

auto metadata_cpu() -> RuntimeMetadata {
    auto metadata = RuntimeMetadata{};
    metadata.device_name = "host";
    metadata.device_vendor = "host";
    metadata.runtime_version = "native";
    metadata.driver_version = "n/a";
    metadata.compiler = compiler_description();
    metadata.cxx_standard = cxx_standard_description();
    return metadata;
}

}  // namespace pgkl
