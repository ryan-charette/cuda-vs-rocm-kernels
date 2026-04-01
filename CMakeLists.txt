cmake_minimum_required(VERSION 3.24)

project(portable_gpu_kernel_lab VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PGKL_ENABLE_CUDA "Build CUDA backend" ON)
option(PGKL_ENABLE_HIP  "Build HIP backend" OFF)

add_library(pgkl_common INTERFACE)
target_include_directories(pgkl_common INTERFACE ${PROJECT_SOURCE_DIR}/include)

add_library(pgkl_cpu STATIC
  src/cpu/reduction_cpu.cpp
  src/cpu/stencil2d_cpu.cpp
  src/cpu/matmul_tiled_cpu.cpp
)
target_link_libraries(pgkl_cpu PUBLIC pgkl_common)

include(CheckLanguage)

if(PGKL_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND DEFINED ENV{CUDAARCHS})
      set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDAARCHS})
    endif()

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      message(WARNING "CMAKE_CUDA_ARCHITECTURES is not set. In Colab, export CUDAARCHS first.")
    else()
      message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    add_library(pgkl_cuda STATIC
      src/cuda/reduction_cuda.cu
      src/cuda/stencil2d_cuda.cu
      src/cuda/matmul_tiled_cuda.cu
    )
    set_target_properties(pgkl_cuda PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(pgkl_cuda PUBLIC pgkl_common)
  else()
    message(WARNING "CUDA compiler not found. Building CPU-only.")
  endif()
endif()

if(PGKL_ENABLE_HIP)
  message(STATUS "HIP scaffold exists, but HIP build is not enabled in this starter CMake.")
endif()

add_subdirectory(bench)
add_subdirectory(tests)
