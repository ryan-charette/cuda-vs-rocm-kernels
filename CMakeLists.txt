cmake_minimum_required(VERSION 3.24)

project(portable_gpu_kernel_lab VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PGKL_ENABLE_CUDA "Build CUDA backend" ON)
option(PGKL_ENABLE_HIP  "Build HIP backend" OFF)
option(PGKL_ENABLE_SYCL "Build SYCL backend" OFF)

add_library(pgkl_common STATIC
  src/common/cli.cpp
)
target_include_directories(pgkl_common PUBLIC ${PROJECT_SOURCE_DIR}/include)

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

    add_library(pgkl_cuda STATIC
      src/cuda/reduction_cuda.cu
      src/cuda/stencil2d_cuda.cu
      src/cuda/matmul_tiled_cuda.cu
    )
    set_target_properties(pgkl_cuda PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(pgkl_cuda PUBLIC pgkl_common)
  endif()
endif()

if(PGKL_ENABLE_HIP)
  message(STATUS "HIP scaffold exists, but HIP build is not enabled in this starter CMake.")
endif()

if(PGKL_ENABLE_SYCL)
  include(CheckCXXCompilerFlag)
  set(PGKL_SYCL_FLAG "-fsycl" CACHE STRING "Compiler flag required to enable SYCL")

  check_cxx_compiler_flag(${PGKL_SYCL_FLAG} PGKL_COMPILER_SUPPORTS_SYCL_FLAG)
  if(PGKL_COMPILER_SUPPORTS_SYCL_FLAG)
    add_library(pgkl_sycl STATIC
      src/sycl/reduction_sycl.cpp
      src/sycl/stencil2d_sycl.cpp
      src/sycl/matmul_tiled_sycl.cpp
    )
    target_compile_options(pgkl_sycl PRIVATE ${PGKL_SYCL_FLAG})
    target_link_options(pgkl_sycl PRIVATE ${PGKL_SYCL_FLAG})
    target_link_libraries(pgkl_sycl PUBLIC pgkl_common)
  else()
    message(WARNING "PGKL_ENABLE_SYCL=ON but compiler does not accept ${PGKL_SYCL_FLAG}; skipping SYCL backend")
  endif()
endif()

add_subdirectory(bench)
add_subdirectory(tests)
