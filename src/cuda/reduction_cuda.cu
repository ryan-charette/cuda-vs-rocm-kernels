#include "pgkl/reduction.hpp"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <vector>

namespace pgkl {
    namespace {

        #define PGKL_CUDA_CHECK(call)
            do {
                cudaError_t err__ = (call);
                if (err__ != cudaSuccess) {
                    std::ostringstream oss;
                    oss << "CUDA error: " << cudaGetErrorString(err__)
                        << " at " << __FILE__ << ":" << __LINE__;
                    throw std::runtime_error(oss.str());
                }
            } while (0)   

    } // namespace

} // namespace pgkl
