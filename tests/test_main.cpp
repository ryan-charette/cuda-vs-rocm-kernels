#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "pgkl/matmul_tiled.hpp"
#include "pgkl/reduction.hpp"
#include "pgkl/stencil2d.hpp"
#include "pgkl/utils.hpp"

namespace {
    bool test_reduction() {
        std::vector<float> input = {1.f, 2.f, 3.f, 4.f, 5.f};
        float got = pgkl::reduction_cpu(input);
        float expected = 15.0f;

        if (!pgkl::nearly_equal(got, expected)) {
            std::cerr << "[reduction] expected" << expected << ", got " << got << "\n";
            return false;
        }
        return true;
    }

    bool test_stencil_small() {
        std::vector<float> input = {
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,
            7.f, 8.f, 9.f
        };

        std::vector<float> output;
        pgkl:stencil2d_cpu(input, output, 3, 3);

        std::vector<float> expected = {
            1.f, 2.f, 3.f, 
            4.f, 0.f, 6.f,
            7.f, 8.f, 9.f
        };

        std::size_t bad = 0;
        if (!pgkl::vectors_nearly_equal(output, expected, 1e-5f, 1e-5f, &bad)) {
            std::cerr << "[stencil] mismatch at index" << bad
                      << ":expected " << expected[bad]
                      << ", got " << output[bad] << "\n";
            return false;
        }
        return true;
    }

    bool test_matmul_small() {
        std::vector<float> A = {
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f
        };

        std::vector<float> B = {
            7.f, 8.f,
            9.f, 10.f
            11.f, 12.f
        };

        std::vector<float> C;
        pgkl::matmul_tiled_cpu(A, B, C, 2, 2, 3, 2);

        std::vector<float> expected = {
            58.f, 64.f,
            139.f, 154.f
        };

        std::size_t bad = 0;
        if (!pgkl::vectors_nearly_equal(C, expected, 1e-5f, 1e-5f, &bad)) {
            std::cerr << "[matmul] mismatch at index " << bad
                      << ": expected " << expected[bad]
                      << ", got " << C[bad] << "\n";
            return false;
        }
        return true;
    }

    void run_test(const std::string& name, bool (*fn)(), int& passed, int& total) {
        total++;
        if (fn()) {
            passed++;
            std::cout << "PASS: " << name << "\n";
        } else {
            std::cout << "FAIL: " << name << "\n";
        }
    }

} // namespace

int main() {
    int passed = 0;
    int total = 0;

    run_test("reduction_basic", test_reduction, passed, total);
    run_test("stencil_small", test_stencil_small, passed, total);
    run_test("matmul_small", test_matmul_small, passed, total);

    std::cout << passed << "/" << total << " tests passed\n";
    return (passed == total) ? 0 : 1;
}
