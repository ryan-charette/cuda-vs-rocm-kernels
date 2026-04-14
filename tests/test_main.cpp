#include "pgkl/matmul_tiled.hpp"
#include "pgkl/reduction.hpp"
#include "pgkl/stencil2d.hpp"
#include "pgkl/utils.hpp"

#ifdef PGKL_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <string>
#include <vector>

namespace {

auto test_reduction_cpu() -> bool {
    const auto input = std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    const auto got = pgkl::reduction_cpu(input);
    constexpr auto expected = 15.0F;

    if (!pgkl::nearly_equal(got, expected)) {
        std::cerr << "[reduction_cpu] expected " << expected << ", got " << got << '\n';
        return false;
    }
    return true;
}

auto test_stencil_cpu() -> bool {
    const auto input = std::vector<float>{
        1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F,
        7.0F, 8.0F, 9.0F,
    };

    auto output = std::vector<float>(input.size(), 0.0F);
    pgkl::stencil2d_cpu(input, output, 3, 3);

    const auto expected = std::vector<float>{
        1.0F, 2.0F, 3.0F,
        4.0F, 0.0F, 6.0F,
        7.0F, 8.0F, 9.0F,
    };

    std::size_t bad_index = 0;
    if (!pgkl::vectors_nearly_equal(output, expected, 1.0e-5F, 1.0e-5F, &bad_index)) {
        std::cerr << "[stencil_cpu] mismatch at index " << bad_index << ": expected " << expected[bad_index]
                  << ", got " << output[bad_index] << '\n';
        return false;
    }
    return true;
}

auto test_matmul_cpu() -> bool {
    const auto a = std::vector<float>{
        1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F,
    };

    const auto b = std::vector<float>{
        7.0F, 8.0F,
        9.0F, 10.0F,
        11.0F, 12.0F,
    };

    auto c = std::vector<float>(4U, 0.0F);
    pgkl::matmul_tiled_cpu(a, b, c, 2, 2, 3, 2);

    const auto expected = std::vector<float>{
        58.0F, 64.0F,
        139.0F, 154.0F,
    };

    std::size_t bad_index = 0;
    if (!pgkl::vectors_nearly_equal(c, expected, 1.0e-5F, 1.0e-5F, &bad_index)) {
        std::cerr << "[matmul_cpu] mismatch at index " << bad_index << ": expected " << expected[bad_index]
                  << ", got " << c[bad_index] << '\n';
        return false;
    }
    return true;
}

#ifdef PGKL_HAS_CUDA
auto cuda_available() -> bool {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

auto test_reduction_cuda() -> bool {
    if (!cuda_available()) {
        std::cout << "SKIP: reduction_cuda (no CUDA device)\n";
        return true;
    }

    const auto input = pgkl::make_patterned_vector(4096);
    const auto expected = pgkl::reduction_cpu(input);
    const auto got = pgkl::reduction_cuda(input);
    if (!pgkl::nearly_equal(got, expected, 1.0e-4F, 1.0e-4F)) {
        std::cerr << "[reduction_cuda] expected " << expected << ", got " << got << '\n';
        return false;
    }
    return true;
}

auto test_stencil_cuda() -> bool {
    if (!cuda_available()) {
        std::cout << "SKIP: stencil_cuda (no CUDA device)\n";
        return true;
    }

    const auto input = pgkl::make_grid(32, 32);
    auto expected = std::vector<float>(input.size(), 0.0F);
    auto got = std::vector<float>(input.size(), 0.0F);

    pgkl::stencil2d_cpu(input, expected, 32, 32);
    pgkl::stencil2d_cuda(input, got, 32, 32);

    std::size_t bad_index = 0;
    if (!pgkl::vectors_nearly_equal(got, expected, 1.0e-5F, 1.0e-5F, &bad_index)) {
        std::cerr << "[stencil_cuda] mismatch at index " << bad_index << ": expected " << expected[bad_index]
                  << ", got " << got[bad_index] << '\n';
        return false;
    }
    return true;
}

auto test_matmul_cuda() -> bool {
    if (!cuda_available()) {
        std::cout << "SKIP: matmul_cuda (no CUDA device)\n";
        return true;
    }

    const auto a = pgkl::make_patterned_vector(32U * 32U);
    const auto b = pgkl::make_patterned_vector(32U * 32U);
    auto expected = std::vector<float>(32U * 32U, 0.0F);
    auto got = std::vector<float>(32U * 32U, 0.0F);

    pgkl::matmul_tiled_cpu(a, b, expected, 32, 32, 32, 16);
    pgkl::matmul_tiled_cuda(a, b, got, 32, 32, 32, 16);

    std::size_t bad_index = 0;
    if (!pgkl::vectors_nearly_equal(got, expected, 1.0e-4F, 1.0e-4F, &bad_index)) {
        std::cerr << "[matmul_cuda] mismatch at index " << bad_index << ": expected " << expected[bad_index]
                  << ", got " << got[bad_index] << '\n';
        return false;
    }
    return true;
}
#endif

void run_test(const std::string& name, bool (*fn)(), int& passed, int& total) {
    ++total;
    if (fn()) {
        ++passed;
        std::cout << "PASS: " << name << '\n';
    } else {
        std::cout << "FAIL: " << name << '\n';
    }
}

}  // namespace

int main() {
    int passed = 0;
    int total = 0;

    run_test("reduction_cpu", test_reduction_cpu, passed, total);
    run_test("stencil_cpu", test_stencil_cpu, passed, total);
    run_test("matmul_cpu", test_matmul_cpu, passed, total);
#ifdef PGKL_HAS_CUDA
    run_test("reduction_cuda", test_reduction_cuda, passed, total);
    run_test("stencil_cuda", test_stencil_cuda, passed, total);
    run_test("matmul_cuda", test_matmul_cuda, passed, total);
#endif

    std::cout << passed << '/' << total << " tests passed\n";
    return passed == total ? 0 : 1;
}
