#pragma once

#include <cstddef>
#include <span>

#include "pgkl/timing.hpp"

namespace pgkl {

void matmul_tiled_cpu(std::span<const float> a,
                      std::span<const float> b,
                      std::span<float> c,
                      std::size_t m,
                      std::size_t n,
                      std::size_t k,
                      std::size_t tile_size = 32,
                      TimingResult* timing = nullptr);

void matmul_tiled_cuda(std::span<const float> a,
                       std::span<const float> b,
                       std::span<float> c,
                       std::size_t m,
                       std::size_t n,
                       std::size_t k,
                       std::size_t tile_size = 32,
                       TimingResult* timing = nullptr);

void matmul_tiled_hip(std::span<const float> a,
                      std::span<const float> b,
                      std::span<float> c,
                      std::size_t m,
                      std::size_t n,
                      std::size_t k,
                      std::size_t tile_size = 32,
                      TimingResult* timing = nullptr);

void matmul_tiled_sycl(std::span<const float> a,
                       std::span<const float> b,
                       std::span<float> c,
                       std::size_t m,
                       std::size_t n,
                       std::size_t k,
                       std::size_t tile_size = 32,
                       TimingResult* timing = nullptr);

}  // namespace pgkl
