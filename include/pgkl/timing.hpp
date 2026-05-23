#pragma once

namespace pgkl {

struct TimingResult {
    double kernel_time_ms{};
    bool kernel_time_available{false};

    void add_kernel_time_ms(const double value) noexcept {
        kernel_time_ms += value;
        kernel_time_available = true;
    }
};

}  // namespace pgkl
