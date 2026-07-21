#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Wrap EVERY cuda API call. Silent failures are the #1 time sink when learning.
#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err_ = (expr);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "[CUDA ERROR] %s:%d '%s' -> %s\n", __FILE__,        \
                   __LINE__, #expr, cudaGetErrorString(err_));                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// Call immediately after every kernel launch. The sync catches async faults
// that would otherwise surface at a confusing place much later.
#define KERNEL_CHECK()                                                         \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)

// Cheap ceil-div. Getting this wrong is the classic off-by-one on awkward sizes.
__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
