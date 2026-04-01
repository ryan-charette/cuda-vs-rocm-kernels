add_executable(pgkl_tests test_main.cpp)
target_link_libraries(pgkl_tests PRIVATE pgkl_cpu pgkl_common)

if(TARGET pgkl_cuda)
  target_link_libraries(pgkl_tests PRIVATE pgkl_cuda)
  target_compile_definitions(pgkl_tests PRIVATE PGKL_HAS_CUDA=1)
endif()
