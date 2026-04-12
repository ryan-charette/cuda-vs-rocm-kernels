#include "pgkl/reduction.hpp"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <vector>

namespace pgkl {

    void check_cuda(cudaError_t err, const std::string& message) {
        if (err != cudaSuccess) {
            throw std::runtime_error(message + ": " + cudaGetErrorString(err));
        }
    }

    } // namespace pgkl
