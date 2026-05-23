# Portable GPU Kernel Lab

This project implements and benchmarks the same kernels across:

- C++ CPU reference code
- CUDA for NVIDIA GPUs
- HIP / ROCm for AMD GPUs
- SYCL for cross-vendor GPU experiments

The goal is to make implementation details, correctness, timing methodology, and performance tradeoffs directly comparable across backends.

---

## Scope

The project focuses on three representative kernels:

1. Reduction
2. 2D stencil
3. Tiled matrix multiply

These kernels cover a useful mix of GPU behavior:

- hierarchical parallel decomposition
- synchronization patterns
- shared memory and local memory use
- global memory access patterns
- memory-bound and compute-bound workloads

---

## Current Status

Implemented:

- CPU reference implementations for all three kernels
- CUDA implementations for all three kernels
- HIP / ROCm implementations for all three kernels
- SYCL implementations for all three kernels
- CMake build switches for CPU, CUDA, HIP, and SYCL
- CLI-driven benchmark runner
- Warmup and repeat controls
- Per-run correctness checks against the CPU reference
- CSV benchmark output with correctness, timing, and device/compiler/runtime metadata
- Initial correctness tests

Still in progress:

- Validation on real NVIDIA hardware
- Validation on real ROCm hardware
- Validation of the selected SYCL compiler/device combinations
- Benchmark sweep scripts that write files under `results/raw/`
- Profiler captures and analysis artifacts
- Roofline-style plots and writeup
- JSON benchmark output

Validation notes:

- CPU builds, tests, and benchmark smoke runs have been verified locally.
- CUDA integration is complete at the source/build level, but still needs validation on NVIDIA hardware.
- HIP integration is complete at the source/build level, but still needs validation on ROCm hardware.
- SYCL integration is complete at the source/build level, but still needs validation with the selected SYCL compiler and target devices.

---

## Repository Layout

```text
.
|-- CMakeLists.txt
|-- bench/
|-- cmake/
|-- docs/
|-- include/
|   `-- pgkl/
|-- results/
|   |-- plots/
|   |-- profiles/
|   `-- raw/
|-- scripts/
|-- src/
|   |-- common/
|   |-- cpu/
|   |-- cuda/
|   |-- hip/
|   `-- sycl/
`-- tests/
```

---

## Build

### CPU-Only

```bash
cmake -S . -B build-cpu -DPGKL_ENABLE_CUDA=OFF -DPGKL_ENABLE_HIP=OFF -DPGKL_ENABLE_SYCL=OFF
cmake --build build-cpu -j
```

On Windows with Visual Studio generators:

```powershell
cmake -S . -B build-cpu -DPGKL_ENABLE_CUDA=OFF -DPGKL_ENABLE_HIP=OFF -DPGKL_ENABLE_SYCL=OFF
cmake --build build-cpu --config Release -j
```

### CUDA

```bash
export CUDAARCHS=native
cmake -S . -B build-cuda -DPGKL_ENABLE_CUDA=ON -DPGKL_ENABLE_HIP=OFF -DPGKL_ENABLE_SYCL=OFF
cmake --build build-cuda -j
```

Set `CUDAARCHS` explicitly if `native` is not supported or if you are cross-building for a known target architecture.

### HIP / ROCm

```bash
export AMDGPU_TARGETS=gfx1100
cmake -S . -B build-hip \
  -DPGKL_ENABLE_CUDA=OFF \
  -DPGKL_ENABLE_HIP=ON \
  -DPGKL_ENABLE_SYCL=OFF \
  -DCMAKE_PREFIX_PATH=/opt/rocm

cmake --build build-hip -j
```

If ROCm is not detected automatically:

```bash
cmake -S . -B build-hip \
  -DPGKL_ENABLE_CUDA=OFF \
  -DPGKL_ENABLE_HIP=ON \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm
```

### SYCL

```bash
cmake -S . -B build-sycl \
  -DPGKL_ENABLE_CUDA=OFF \
  -DPGKL_ENABLE_HIP=OFF \
  -DPGKL_ENABLE_SYCL=ON \
  -DCMAKE_CXX_COMPILER=icpx

cmake --build build-sycl -j
```

If your compiler requires a different enable flag, override `PGKL_SYCL_FLAG`:

```bash
cmake -S . -B build-sycl \
  -DPGKL_ENABLE_SYCL=ON \
  -DPGKL_SYCL_FLAG=-fsycl \
  -DCMAKE_CXX_COMPILER=icpx
```

You can combine backends in one build when the selected compiler/toolchain supports them, for example:

```bash
cmake -S . -B build-gpu -DPGKL_ENABLE_CUDA=ON -DPGKL_ENABLE_SYCL=ON
cmake --build build-gpu -j
```

---

## Run

### Tests

```bash
./build-cpu/tests/pgkl_tests
```

With Visual Studio generators:

```powershell
.\build-cpu\tests\Release\pgkl_tests.exe
```

### Benchmarks

Reduction:

```bash
./build-cpu/bench/pgkl_bench \
  --backend cpu \
  --kernel reduction \
  --size 1048576 \
  --repeats 5 \
  --warmups 1
```

2D stencil:

```bash
./build-cpu/bench/pgkl_bench \
  --backend cpu \
  --kernel stencil2d \
  --size 1024 \
  --repeats 5 \
  --warmups 1
```

Tiled matrix multiply:

```bash
./build-cpu/bench/pgkl_bench \
  --backend cpu \
  --kernel matmul \
  --size 512 \
  --tile-size 32 \
  --repeats 5 \
  --warmups 1
```

Use `--backend cpu|cuda|hip|sycl` to select the backend. Correctness is checked on every measured repeat by default; pass `--skip-correctness` only when intentionally timing without validation.

Example CSV run:

```bash
./build-cuda/bench/pgkl_bench \
  --backend cuda \
  --kernel matmul \
  --size 1024 \
  --tile-size 32 \
  --repeats 10 \
  --warmups 3 \
  --format csv
```

CSV output includes:

- `avg_end_to_end_ms`: host-observed runtime for the backend function, including setup, copies, launch, and synchronization work inside that function
- `avg_kernel_ms`: device event/profiling time for launched kernels; for CPU this is the measured function time
- `kernel_timing_available`: whether `avg_kernel_ms` came from a complete set of backend timing samples
- `correct`: whether every checked repeat matched the CPU reference
- device name, vendor, runtime version, driver version, compiler, and C++ standard metadata

---

## Suggested Hardware Validation Flow

On each machine:

1. Record environment details in `docs/environment.md`.
2. Configure and build the relevant backend.
3. Run `pgkl_tests`.
4. Run a small benchmark with correctness enabled.
5. Run larger benchmark sweeps with `--format csv`.
6. Save raw CSV output under `results/raw/`.
7. Capture profiler data under `results/profiles/` when profiling tools are available.

Use the same problem sizes, repeat counts, warmups, and tile sizes across backends when comparing results.

---

## Measurement Roadmap

Planned additions:

- benchmark sweep scripts that write CSV files under `results/raw/`
- profiler captures with Nsight, rocprof, and/or Omniperf
- kernel-level metrics such as occupancy, bandwidth, and achieved FLOPs
- roofline-style analysis
- plots comparing CPU, CUDA, HIP, and SYCL runs
- JSON benchmark output
