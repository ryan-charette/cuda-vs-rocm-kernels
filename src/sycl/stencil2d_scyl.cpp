#include "pgkl/stencil2d.hpp"

#include "sycl_compat.hpp"

#include <stdexcept>
#include <string>

namespace pgkl {

void stencil2d_sycl(const std::span<const float> input,
                    std::span<float> output,
                    const std::size_t rows,
                    const std::size_t cols) {
    if ((rows * cols) != input.size()) {
        throw std::invalid_argument("stencil2d_sycl: rows * cols must equal input.size()");
    }
    if (output.size() != input.size()) {
        throw std::invalid_argument("stencil2d_sycl: output.size() must equal input.size()");
    }

    namespace sycl = pgkl::sycl_compat;

    try {
        sycl::queue queue{sycl::default_selector_v};
        sycl::buffer<float> input_buffer(input.data(), sycl::range<1>(input.size()));
        sycl::buffer<float> output_buffer(output.data(), sycl::range<1>(output.size()));

        const std::size_t tile_rows = 16U;
        const std::size_t tile_cols = 16U;
        const std::size_t global_rows = ((rows + tile_rows - 1U) / tile_rows) * tile_rows;
        const std::size_t global_cols = ((cols + tile_cols - 1U) / tile_cols) * tile_cols;

        queue.submit([&](sycl::handler& cgh) {
            auto in = input_buffer.template get_access<sycl::access::mode::read>(cgh);
            auto out = output_buffer.template get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(global_rows, global_cols), sycl::range<2>(tile_rows, tile_cols)),
                [=](sycl::nd_item<2> item) {
                    const std::size_t row = item.get_global_id(0);
                    const std::size_t col = item.get_global_id(1);

                    if (row >= rows || col >= cols) {
                        return;
                    }

                    const std::size_t index = (row * cols) + col;
                    if (row == 0U || col == 0U || row + 1U == rows || col + 1U == cols) {
                        out[index] = in[index];
                        return;
                    }

                    const float up = in[((row - 1U) * cols) + col];
                    const float down = in[((row + 1U) * cols) + col];
                    const float left = in[index - 1U];
                    const float right = in[index + 1U];
                    const float center = in[index];
                    out[index] = up + down + left + right - (4.0F * center);
                });
        });
        queue.wait_and_throw();
    } catch (const sycl::exception& error) {
        throw std::runtime_error(std::string{"stencil2d_sycl failed: "} + error.what());
    }
}

}  // namespace pgkl
