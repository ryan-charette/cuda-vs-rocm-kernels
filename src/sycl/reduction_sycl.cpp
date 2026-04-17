#include "pgkl/reduction.hpp"

#include "sycl_compat.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace pgkl {
namespace {

constexpr std::size_t kWorkgroupSize = 256U;

[[nodiscard]] auto next_multiple(const std::size_t value, const std::size_t factor) -> std::size_t {
    return ((value + factor - 1U) / factor) * factor;
}

}  // namespace

auto reduction_sycl(const std::span<const float> input) -> float {
    if (input.empty()) {
        return 0.0F;
    }

    namespace sycl = pgkl::sycl_compat;

    try {
        sycl::queue queue{sycl::default_selector_v};
        auto current = std::vector<float>(input.begin(), input.end());

        while (current.size() > 1U) {
            const std::size_t partial_count = (current.size() + kWorkgroupSize - 1U) / kWorkgroupSize;
            const std::size_t global_size = next_multiple(partial_count * kWorkgroupSize, kWorkgroupSize);
            auto partial = std::vector<float>(partial_count, 0.0F);

            {
                sycl::buffer<float> input_buffer(current.data(), sycl::range<1>(current.size()));
                sycl::buffer<float> partial_buffer(partial.data(), sycl::range<1>(partial.size()));

                queue.submit([&](sycl::handler& cgh) {
                    auto in = input_buffer.template get_access<sycl::access::mode::read>(cgh);
                    auto out = partial_buffer.template get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::local_accessor<float, 1> scratch(sycl::range<1>(kWorkgroupSize), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kWorkgroupSize)),
                        [=](sycl::nd_item<1> item) {
                            const auto local_id = item.get_local_id(0);
                            const auto group_id = item.get_group(0);
                            const auto index = item.get_global_id(0);

                            scratch[local_id] = index < in.size() ? in[index] : 0.0F;
                            item.barrier(sycl::access::fence_space::local_space);

                            for (std::size_t stride = kWorkgroupSize / 2U; stride > 0U; stride >>= 1U) {
                                if (local_id < stride) {
                                    scratch[local_id] += scratch[local_id + stride];
                                }
                                item.barrier(sycl::access::fence_space::local_space);
                            }

                            if (local_id == 0U && group_id < out.size()) {
                                out[group_id] = scratch[0];
                            }
                        });
                });
                queue.wait_and_throw();
            }

            current = std::move(partial);
        }

        return current.front();
    } catch (const sycl::exception& error) {
        throw std::runtime_error(std::string{"reduction_sycl failed: "} + error.what());
    }
}

}  // namespace pgkl
