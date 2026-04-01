# Portable GPU Kernel Lab: CUDA vs ROCm vs CPU Roofline

<<<<<<< HEAD
## Project status

Current scope is locked to three kernels across CPU, CUDA, and HIP/ROCm:
=======
A benchmarking and analysis repo for comparing the same core kernels across:

- CPU baseline
- NVIDIA GPU via CUDA
- AMD GPU via HIP/ROCm

## Locked kernel set
>>>>>>> de3beaa (Scaffold repo structure and Colab-friendly CMake setup)

- reduction
- 2D stencil
- tiled matrix multiply

<<<<<<< HEAD
The project focuses on correctness, benchmarking, profiling, roofline-style 
=======
## Project focus

This project is about more than kernel code. The final repo should demonstrate:

- correctness across backends
- fair benchmarking methodology
- profiling artifacts and explanation
- roofline-style performance analysis
- reproducible setup

## Planned architecture

```text
.
├── CMakeLists.txt
├── README.md
├── cmake/
├── include/
│   └── pgkl/
├── src/
│   ├── common/
│   ├── cpu/
│   ├── cuda/
│   └── hip/
├── bench/
├── tests/
├── scripts/
├── results/
│   ├── raw/
│   ├── plots/
│   └── profiles/
└── docs/

```

## Backend plan

### CPU
Reference implementations in C++ used for correctness and baseline timing.

### CUDA
Primary GPU implementation path for development in Google Colab.

### HIP/ROCm
Scaffolded now, to be implemented and validated later on AMD hardware or a ROCm-capable environment.

## Build plan

### Colab / CUDA path
Use a GPU runtime, detect the GPU compute capability, export `CUDAARCHS`, then configure with CMake.

### CPU-only fallback
If `nvcc` is unavailable, configure with `-DPGKL_ENABLE_CUDA=OFF`.

## Near-term milestones

1. Day 2: repo scaffold, CMake, README, placeholder source files
2. Day 3+: CPU baseline implementations
3. Later: CUDA kernels, benchmark harness, profiling, analysis
>>>>>>> de3beaa (Scaffold repo structure and Colab-friendly CMake setup)
