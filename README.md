# Portable GPU Kernel Lab

This project is a cross-backend kernel benchmarking project that implements and evaluates the same workloads across:

- **C++ (CPU reference)**
- **CUDA (NVIDIA GPUs)**
- **HIP / ROCm (AMD GPUs)**

The repository is structured to make implementation details, performance characteristics, and tradeoffs directly comparable across backends.

---

## Scope

The project focuses on three representative kernels:

1. **Reduction**
2. **2D stencil**
3. **Tiled matrix multiply**

These kernels were selected to cover a meaningful range of behaviors:

- hierarchical parallel decomposition (thread / block / grid)
- synchronization patterns
- shared memory vs global memory usage
- memory access patterns and locality
- compute vs memory bound workloads

---

## Current status

### Implemented

- CPU reference implementations:
  - reduction
  - 2D stencil
  - tiled matrix multiply
- CUDA implementations for all three kernels
- HIP / ROCm implementations for all three kernels
- CMake-based multi-backend build system
- CLI-driven benchmark runner
- Initial correctness tests

### In progress

- Full validation on ROCm hardware
- Additional validation on NVIDIA hardware
- Structured benchmark output (CSV / JSON)
- Profiling integration and analysis artifacts
- Automation scripts for repeatable runs

---

## Validation notes

- CPU builds and runs have been verified
- CUDA code compiles, but requires additional validation on target hardware
- HIP integration is complete at the source level, but has not yet been validated on a ROCm system in this environment

---

## Repository layout

```text
.
├── CMakeLists.txt
├── bench/
├── cmake/
├── docs/
├── include/
│   └── pgkl/
├── results/
│   ├── plots/
│   ├── profiles/
│   └── raw/
├── scripts/
├── src/
│   ├── common/
│   ├── cpu/
│   ├── cuda/
│   └── hip/
└── tests/
```

---

## Build

### CPU-only

```bash
cmake -S . -B build -DPGKL_ENABLE_CUDA=OFF
cmake --build build -j
```

### CUDA

```bash
export CUDAARCHS=native
cmake -S . -B build
cmake --build build -j
```

If needed, set `CUDAARCHS` explicitly for your target GPU.

### HIP / ROCm

```bash
cmake -S . -B build \
  -DPGKL_ENABLE_CUDA=OFF \
  -DPGKL_ENABLE_HIP=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm

cmake --build build -j
```

If ROCm is not detected automatically:

```bash
-D CMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm
```

---

## Run

### Tests

```bash
./build/tests/pgkl_tests
```

### Benchmarks

Reduction:

```bash
./build/bench/pgkl_bench \
  --backend cpu \
  --kernel reduction \
  --size 1048576 \
  --repeats 5
```

Stencil:

```bash
./build/bench/pgkl_bench \
  --backend cpu \
  --kernel stencil2d \
  --size 1024 \
  --repeats 5
```

Matrix multiply:

```bash
./build/bench/pgkl_bench \
  --backend cpu \
  --kernel matmul \
  --size 512 \
  --tile-size 32 \
  --repeats 5
```

Current output is printed to stdout. Structured result export is being added.

---

## Measurement roadmap

Planned additions:

- structured benchmark outputs under `results/raw/`
- profiler captures (Nsight / rocprof)
- kernel-level metrics (occupancy, bandwidth, achieved FLOPs)
- roofline-style analysis
- comparable runs across CPU / CUDA / HIP
