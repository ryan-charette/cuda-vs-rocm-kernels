add_executable(pgkl_bench benchmark_main.cpp)
target_link_libraries(pgkl_bench PRIVATE pgkl_cpu pgkl_common)

if(TARGET pgkl_cuda)
  target_link_libraries(pgkl_bench PRIVATE pgkl_cuda)
endif()
