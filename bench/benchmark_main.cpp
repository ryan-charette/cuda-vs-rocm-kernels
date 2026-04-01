#include <iostream>

int main() {
  std::cout << "pgkl_bench: benchmark harness scaffold" << std::endl;
#ifdef PGKL_HAS_CUDA
  std::cout << "CUDA target is enabled in this build." << std::endl;
#else
  std::cout << "CUDA target is not enabled in this build." << std::endl;
#endif
  return 0;
}
