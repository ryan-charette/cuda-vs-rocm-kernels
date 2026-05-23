#pragma once

#include "pgkl/timing.hpp"
#include "sycl_compat.hpp"

#include <cstdint>

namespace pgkl::sycl_detail {

[[nodiscard]] inline auto make_queue(TimingResult* timing) -> sycl_compat::queue {
    if (timing == nullptr) {
        return sycl_compat::queue{sycl_compat::default_selector_v};
    }

    return sycl_compat::queue{
        sycl_compat::default_selector_v,
        sycl_compat::property_list{sycl_compat::property::queue::enable_profiling{}},
    };
}

inline void add_event_kernel_time(const sycl_compat::event& event, TimingResult* timing) {
    if (timing == nullptr) {
        return;
    }

    const auto start_ns =
        event.get_profiling_info<sycl_compat::info::event_profiling::command_start>();
    const auto end_ns =
        event.get_profiling_info<sycl_compat::info::event_profiling::command_end>();
    timing->add_kernel_time_ms(static_cast<double>(end_ns - start_ns) / 1.0e6);
}

}  // namespace pgkl::sycl_detail
