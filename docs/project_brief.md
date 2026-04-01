# Portable GPU Kernel Lab: CUDA vs ROCm vs CPU Roofline

## Project brief

### Summary
This project is a portable GPU benchmarking lab that compares the same core kernels across three execution targets:

- CPU baseline
- NVIDIA GPU via CUDA
- AMD GPU via HIP/ROCm

The goal is to demonstrate practical understanding of GPU computing at the kernel level: execution hierarchy, memory movement, synchronization, profiling, benchmarking methodology, and performance portability across vendors.

This is primarily an engineering and performance analysis project, not just an implementation exercise. The focus is on building correct implementations, measuring them fairly, profiling their behavior, and explaining why performance differs across backends.

This project is designed as a portfolio piece for systems/HPC roles and as a structured platform for learning cross-vendor GPU performance.

---

## Goals

The project should answer these questions:

1. How do common HPC-style kernels behave on CPU, CUDA, and HIP/ROCm?
2. Which kernels are memory-bound versus compute-bound in practice?
3. What implementation choices matter most for performance?
4. How portable are kernel designs across NVIDIA and AMD backends?
5. How closely do measured results match a simple roofline-style performance model?

---

## Kernel set

This project will implement and benchmark exactly these three kernels:

1. **Reduction**
   - Representative of parallel aggregation
   - Good for studying synchronization, thread/block structure, and memory bandwidth

2. **2D stencil**
   - Representative of neighborhood-based grid computation
   - Good for studying locality, cache/shared memory behavior, and memory-bound performance

3. **Tiled matrix multiply**
   - Representative of dense linear algebra
   - Good for studying arithmetic intensity, tiling, shared memory, and compute-bound behavior

These kernels provide a balanced view of GPU programming while keeping scope controlled.

---

## Backends in scope

### Required
- CPU reference implementation in C++ for correctness
- CUDA implementation for NVIDIA GPUs
- HIP/ROCm implementation for AMD GPUs

### Optional stretch
- Optimized CPU baseline implementation (e.g., Open MP and/or SIMD) for fairer comparison
- Triton implementation for one kernel only, if time allows

---

## Benchmark scale

Benchmarks should include multiple problem sizes:

- Small: fast correctness validation (e.g., 1e4-1e5 operations)
- Medium: cache-relavant sizes (e.g., 1e6-1e7 operations)
- Large: GPU-saturating workloads (e.g, 1e8+ operations)

All backends should use identical problem sizes for fair comparison.

---

## Fair comparison principles

To ensure meaningful results

- Use the same algorithmic structure across backends
- Separate kernel execution time from data transfer time
- Use consistent timing methodology (warmups, repetitions, synchronization)
- Clearly document any backend-specific optimizations

---

## Deliverables

The final repo should contain:

### 1. Kernel implementations
- CPU, CUDA, and HIP versions of each kernel
- Shared problem setup and output validation logic
- Clear separation between backend-specific code and common utilities

### 2. Correctness tests
- Tests that compare outputs across all supported backends
- Small and medium problem sizes for fast validation
- Tolerance-based comparison for floating-point results where needed

### 3. Benchmark harness
- CLI or scriptable benchmark runner
- Separate reporting for:
  - kernel execution time
  - transfer time where applicable
  - end-to-end time
- Repeated runs with warmup and summary statistics
- Export of benchmark data to CSV or JSON

### 4. Profiling artifacts
- Nsight Compute and/or Nsight Systems captures for CUDA runs
- rocprof and/or Omniperf captures for HIP/ROCm runs when available
- Saved screenshots and short written explanations of what they show
- Clear linkage between profiler data and performance results

### 5. Roofline-style analysis
- Approximate arithmetic intensity estimates per kernel
- Effective bandwidth and/or throughput estimates from benchmark results
- Estimation of:
  - peak memory bandwidth (e.g., memcpy benchmark or specs)
  - peak compute throughput (based on hardware specs)
- Plots that position kernels on a simplified roofline chart
- Discussion of why each kernel appears memory- or compute-limited

### 6. Reproducibility setup
- CMake-based build system
- Docker and/or Singularity/Apptainer setup where practical
- Scripts for building, testing, benchmarking, and plotting
- Environment notes with compiler/tool versions and hardware details

### 7. Technical writeup
- README that explains design, methodology, results, and limitations
- Clear discussion of performance observations

---

## Success criteria

The project is successful only if all of the following are true.

### Correctness across backends
- Each kernel produces results that match the CPU reference within defined tolerances
- Correctness tests run reliably across supported backends
- Output mismatches are easy to diagnose

### Benchmark harness
- Benchmarks run from a consistent interface
- Problem sizes, iteration counts, and warmups are configurable
- Results are exported in a format that can be plotted and reproduced
- Timing methodology is documented and applied consistently

### Profiling artifacts
- At least one meaningful profiler capture exists for each required kernel on CUDA
- At least one meaningful profiler capture exists for HIP/ROCm where tooling is available
- Screenshots are stored in the repo and explained in writing
- The analysis links profiler observations to performance results

### Roofline-style analysis
- Each kernel includes a rough arithmetic-intensity estimate
- Result plots compare achieved behavior against a simplified compute/bandwidth model
- The writeup explains whether each kernel is primarily bandwidth-limited or compute-limited

### Reproducible setup
- A new user can build and run the project by following the documented steps
- Build, test, benchmark, and plot commands are scripted
- Required tool versions, GPU targets, and environment assumptions are documented

---

## Known challenges

- Differences between CUDA and HIP programming models
- Tooling asymmetry (Nsight vs ROCm tools)
- Ensuring apples-to-apples benchmarking across hardware
- Managing memory transfer vs compute separation correctly

---

## Non-goals

To keep the project focused, the following are explicitly out of scope for the initial version:

- Implementing more than three required kernels
- Exhaustive vendor-specific micro-optimization
- Matching cuBLAS/rocBLAS-class performance
- Multi-GPU execution
- Distributed training or cluster-scale benchmarking
- Supporting every GPU architecture

This project prioritizes clarity, fairness, portability, and analysis over maximum absolute performance.

---

## Technical approach

### Language and build
- C++ for implementations
- CMake for build configuration
- Python only for experiment orchestration and plotting

### NVIDIA-side tools
- CUDA Toolkit
- Nsight Compute
- Nsight Systems

### AMD-side tools
- HIP/ROCm
- hipify where useful
- rocprof and/or Omniperf when available
- ROCm container images if practical

---

## Proposed repo structure

```text
/
├── CMakeLists.txt
├── README.md
├── cmake/
├── include/
├── src/
│   ├── cpu/
│   ├── cuda/
│   └── hip/
├── tests/
├── bench/
├── scripts/
├── results/
│   ├── raw/
│   ├── plots/
│   └── profiles/
└── docs/
    ├── project_brief.md
    ├── environment.md
    └── analysis_notes.md
```

---

## Evaluation questions

By the end of the project, the repo should let a reviewer answer:

- Are the implementations correct?
- Are the benchmarks fair and repeatable?
- Does the author understand the memory and execution behavior of each kernel?
- Can the author explain performance differences between CPU, CUDA, and HIP?
- Can another engineer reproduce the results?

If the answer to all of these questions is yes, then the project has met its purpose.

---

## Definition of done

This project is done when:

- All three kernels run on CPU, CUDA, and HIP/ROCm
- Correctness tests pass across supported backends
- Benchmark runs produce clean exportable data
- Profiler artifacts and screenshots are saved and documented
- Roofline-style plots and analysis are included
- The repo can be built and used from documented steps
- The README is strong enough to serve as a portfolio artifact