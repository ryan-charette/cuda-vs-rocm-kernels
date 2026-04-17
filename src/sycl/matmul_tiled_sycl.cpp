#include "pgkl/matmul_tiled.hpp"

#include "sycl_compat.hpp"

#include <stdexcept>
#include <string>

namespace pgkl {
namespace {

template <std::size_t TileSize>
void run_matmul_kernel(const std::span<const float> a,
                       const std::span<const float> b,
                       std::span<float> c,
                       const std::size_t m,
                       const std::size_t n,
                       const std::size_t k) {
    namespace sycl = pgkl::sycl_compat;

    sycl::queue queue{sycl::default_selector_v};
    sycl::buffer<float> a_buffer(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float> b_buffer(b.data(), sycl::range<1>(b.size()));
    sycl::buffer<float> c_buffer(c.data(), sycl::range<1>(c.size()));

    const std::size_t global_rows = ((m + TileSize - 1U) / TileSize) * TileSize;
    const std::size_t global_cols = ((n + TileSize - 1U) / TileSize) * TileSize;

    queue.submit([&](sycl::handler& cgh) {
        auto acc_a = a_buffer.template get_access<sycl::access::mode::read>(cgh);
        auto acc_b = b_buffer.template get_access<sycl::access::mode::read>(cgh);
        auto acc_c = c_buffer.template get_access<sycl::access::mode::discard_write>(cgh);
        sycl::local_accessor<float, 2> tile_a(sycl::range<2>(TileSize, TileSize), cgh);
        sycl::local_accessor<float, 2> tile_b(sycl::range<2>(TileSize, TileSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(global_rows, global_cols), sycl::range<2>(TileSize, TileSize)),
            [=](sycl::nd_item<2> item) {
                const std::size_t row = item.get_global_id(0);
                const std::size_t col = item.get_global_id(1);
                const std::size_t local_row = item.get_local_id(0);
                const std::size_t local_col = item.get_local_id(1);
                const std::size_t tile_count = (k + TileSize - 1U) / TileSize;

                float sum = 0.0F;
                for (std::size_t tile = 0; tile < tile_count; ++tile) {
                    const std::size_t tiled_col = (tile * TileSize) + local_col;
                    const std::size_t tiled_row = (tile * TileSize) + local_row;

                    tile_a[local_row][local_col] = (row < m && tiled_col < k) ? acc_a[(row * k) + tiled_col] : 0.0F;
                    tile_b[local_row][local_col] = (tiled_row < k && col < n) ? acc_b[(tiled_row * n) + col] : 0.0F;

                    item.barrier(sycl::access::fence_space::local_space);
                    for (std::size_t inner = 0; inner < TileSize; ++inner) {
                        sum += tile_a[local_row][inner] * tile_b[inner][local_col];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (row < m && col < n) {
                    acc_c[(row * n) + col] = sum;
                }
            });
    });
    queue.wait_and_throw();
}

}  // namespace

void matmul_tiled_sycl(const std::span<const float> a,
                       const std::span<const float> b,
                       std::span<float> c,
                       const std::size_t m,
                       const std::size_t n,
                       const std::size_t k,
                       const std::size_t tile_size) {
    if (tile_size == 0U) {
        throw std::invalid_argument("matmul_tiled_sycl: tile_size must be greater than zero");
    }
    if (a.size() != (m * k)) {
        throw std::invalid_argument("matmul_tiled_sycl: a.size() must equal m * k");
    }
    if (b.size() != (k * n)) {
        throw std::invalid_argument("matmul_tiled_sycl: b.size() must equal k * n");
    }
    if (c.size() != (m * n)) {
        throw std::invalid_argument("matmul_tiled_sycl: c.size() must equal m * n");
    }

    try {
        switch (tile_size) {
            case 8U:
                run_matmul_kernel<8U>(a, b, c, m, n, k);
                break;
            case 16U:
                run_matmul_kernel<16U>(a, b, c, m, n, k);
                break;
            case 32U:
                run_matmul_kernel<32U>(a, b, c, m, n, k);
                break;
            default:
                throw std::invalid_argument("matmul_tiled_sycl: supported tile sizes are 8, 16, and 32");
        }
    } catch (const sycl_compat::exception& error) {
        throw std::runtime_error(std::string{"matmul_tiled_sycl failed: "} + error.what());
    }
}

}  // namespace pgkl
