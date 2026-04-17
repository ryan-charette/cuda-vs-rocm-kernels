#pragma once

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
namespace pgkl::sycl_compat = sycl;
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
namespace pgkl::sycl_compat = cl::sycl;
#else
#error "SYCL headers not found. Install a SYCL implementation or disable PGKL_ENABLE_SYCL."
#endif
