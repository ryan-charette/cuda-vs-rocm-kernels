add_executable(pgkl_bench benchmark_main.cpp)
target_link_libraries(pgkl_bench PRIVATE pgkl_cpu pgkl_common)

if(TARGET pgkl_cuda)
  target_link_libraries(pgkl_bench PRIVATE pgkl_cuda)
  target_compile_definitions(pgkl_bench PRIVATE PGKL_HAS_CUDA=1)
endif()

if(TARGET pgkl_sycl)
  target_link_libraries(pgkl_bench PRIVATE pgkl_sycl)
  target_compile_definitions(pgkl_bench PRIVATE PGKL_HAS_SYCL=1)
endif()
