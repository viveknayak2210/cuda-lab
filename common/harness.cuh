#pragma once
#include "check.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <string>
#include <vector>

// Injected by the Makefile (-DLAB_GIT_SHA=...) so every CSV row records which
// source version produced it. "unknown" when built outside the repo.
#ifndef LAB_GIT_SHA
#define LAB_GIT_SHA "unknown"
#endif

namespace lab {

// ---------------------------------------------------------------------------
// The awkward-size battery. Every kernel gets run against all of these before
// it is allowed to be "correct". Covers: empty, single, sub-warp, exact warp,
// warp+1, exact block, block+1, multi-block ragged, and a large prime.
// ---------------------------------------------------------------------------
inline const std::vector<int>& awkward_sizes() {
  static const std::vector<int> v = {
      0,    1,     2,     31,    32,     33,     63,     64,     65,
      127,  128,   129,   255,   256,    257,    511,    512,    513,
      1023, 1024,  1025,  4095,  4096,   4097,   100000, 1048573 /* prime */,
      1 << 22};
  return v;
}

// ---------------------------------------------------------------------------
// Timing: warmup, then N repeats, report the median (not the mean -- one
// scheduling hiccup should not move your number).
// ---------------------------------------------------------------------------
inline float time_kernel_ms(const std::function<void()>& launch, int warmup = 5,
                            int repeats = 50) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> samples;
  samples.reserve(repeats);
  for (int i = 0; i < repeats; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

// ---------------------------------------------------------------------------
// Correctness. Relative error, never exact equality -- these are floats and
// the GPU will legitimately reassociate.
// ---------------------------------------------------------------------------
inline bool compare(const float* got, const float* want, long long n,
                    float rtol = 1e-5f, float atol = 1e-6f,
                    int max_report = 5) {
  long long bad = 0;
  for (long long i = 0; i < n; ++i) {
    float diff = std::fabs(got[i] - want[i]);
    float tol = atol + rtol * std::fabs(want[i]);
    // A NaN in `got` fails !(diff <= tol) on its own (NaN compares false), so
    // no separate isnan test; a NaN that MATCHES a reference NaN is correct.
    if (!(diff <= tol) && !(std::isnan(got[i]) && std::isnan(want[i]))) {
      if (bad < max_report) {
        std::fprintf(stderr, "  mismatch @%lld: got %.9g want %.9g (|d|=%.3g)\n",
                     i, got[i], want[i], diff);
      }
      ++bad;
    }
  }
  if (bad) std::fprintf(stderr, "  %lld/%lld elements wrong\n", bad, n);
  return bad == 0;
}

// ---------------------------------------------------------------------------
// Achievable bandwidth probe. Compare against THIS, not the spec-sheet number:
// the spec number is unreachable and makes every kernel look worse than it is.
//
// A baseline must be >= anything measured against it, so probe BOTH common
// stream mixes (1R+1W scale, 2R+1W triad -- read/write ratio shifts what the
// memory system sustains) across several grid sizes, and keep the best.
// Cached to results/peak_bw.txt keyed by GPU name, so a different card on the
// next pod re-measures automatically instead of poisoning every %-of-peak.
// ---------------------------------------------------------------------------
// The probe buffers must hold incompressible data: Ampere+ can compress
// uniform traffic (e.g. all-zeros) in L2/DRAM, which inflates the measured
// "peak" and makes every real kernel's %-of-peak read low. Cheap integer hash
// per element, done on-device so no 256 MB host allocations.
__global__ void _bw_fill(float* __restrict__ dst, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    unsigned h = (unsigned)i * 2654435761u;  // Knuth multiplicative hash
    dst[i] = (float)(h & 0xFFFF) * (1.f / 65536.f) - 0.5f;
  }
}

__global__ void _bw_scale(float* __restrict__ dst, const float* __restrict__ src,
                          size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) dst[i] = src[i] * 2.0f;
}

__global__ void _bw_triad(float* __restrict__ dst, const float* __restrict__ a,
                          const float* __restrict__ b, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) dst[i] = a[i] + 2.0f * b[i];
}

inline float measure_peak_bw_gbs() {
  int dev = 0;
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

  // Cache format: "<gbs>\n<gpu name>\n". A bare number (the old format) or a
  // name mismatch both fall through to a fresh measurement.
  if (FILE* f = std::fopen("results/peak_bw.txt", "r")) {
    float cached = 0.f;
    char name[256] = {0};
    int ok = std::fscanf(f, "%f ", &cached);
    char* got = std::fgets(name, sizeof(name), f);
    std::fclose(f);
    if (ok == 1 && cached > 0.f && got) {
      name[std::strcspn(name, "\n")] = 0;
      if (std::strcmp(name, p.name) == 0) return cached;
    }
  }

  const size_t n = 1ull << 26;  // 256 MB per buffer
  float *a = nullptr, *b = nullptr, *c = nullptr;
  CUDA_CHECK(cudaMalloc(&a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c, n * sizeof(float)));

  const int block = 256;
  const int fill_grid = p.multiProcessorCount * 32;
  _bw_fill<<<fill_grid, block>>>(a, n);
  _bw_fill<<<fill_grid, block>>>(b, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float best = 0.f;
  for (int mult : {16, 32, 64}) {
    const int grid = p.multiProcessorCount * mult;
    float ms = time_kernel_ms([&] { _bw_scale<<<grid, block>>>(c, a, n); }, 3, 10);
    best = std::max(best, float(2.0 * n * sizeof(float)) / (ms * 1e6f));
    ms = time_kernel_ms([&] { _bw_triad<<<grid, block>>>(c, a, b, n); }, 3, 10);
    best = std::max(best, float(3.0 * n * sizeof(float)) / (ms * 1e6f));
  }
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));

  if (FILE* f = std::fopen("results/peak_bw.txt", "w")) {
    std::fprintf(f, "%.3f\n%s\n", best, p.name);
    std::fclose(f);
  }
  return best;
}

// ---------------------------------------------------------------------------
// One row of the results CSV. Append-only so runs accumulate across sessions.
// The trailing provenance columns (gpu, sm, sha, date) are what let you trust
// a row three pods and six weeks later. They come last so old 7-column rows
// still line up.
// ---------------------------------------------------------------------------
inline void record(const std::string& kernel, const std::string& variant,
                   long long n, float ms, double bytes_moved,
                   double flops = 0.0) {
  FILE* f = std::fopen("results/bench.csv", "a");
  if (!f) return;
  std::fseek(f, 0, SEEK_END);
  if (std::ftell(f) == 0)
    std::fprintf(f, "kernel,variant,n,ms,gbs,gflops,pct_peak_bw,gpu,sm,sha,date\n");

  static float peak = measure_peak_bw_gbs();
  static cudaDeviceProp prop = [] {
    cudaDeviceProp q{};
    int d = 0;
    CUDA_CHECK(cudaGetDevice(&d));
    CUDA_CHECK(cudaGetDeviceProperties(&q, d));
    return q;
  }();
  char date[16];
  std::time_t t = std::time(nullptr);
  std::strftime(date, sizeof(date), "%Y-%m-%d", std::localtime(&t));

  double gbs = bytes_moved / (ms * 1e6);
  double gflops = flops / (ms * 1e6);
  std::fprintf(f, "%s,%s,%lld,%.6f,%.2f,%.2f,%.1f,%s,%d%d,%s,%s\n",
               kernel.c_str(), variant.c_str(), n, ms, gbs, gflops,
               100.0 * gbs / peak, prop.name, prop.major, prop.minor,
               LAB_GIT_SHA, date);
  std::fclose(f);

  std::printf("  %-14s n=%-9lld %8.4f ms  %7.1f GB/s  %5.1f%% of peak\n",
              variant.c_str(), n, ms, gbs, 100.0 * gbs / peak);
}

inline void print_device_banner() {
  int dev = 0;
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  std::printf("== %s | sm_%d%d | %d SMs | %.1f GB | %zu KB smem/block ==\n",
              p.name, p.major, p.minor, p.multiProcessorCount,
              p.totalGlobalMem / 1e9, p.sharedMemPerBlockOptin / 1024);
}

// ---------------------------------------------------------------------------
// Occupancy: closes the loop from `ptxas -v`. Registers/thread and smem/block
// are the INPUTS; this prints the OUTPUT that matters -- how many blocks each
// SM can actually hold at this block size, and what fraction of the SM's
// thread capacity that is. Pass dyn_smem if the kernel launches with dynamic
// shared memory. Costs nothing: pure driver arithmetic, no kernel launch.
// ---------------------------------------------------------------------------
template <typename Kernel>
inline void report_occupancy(const char* name, Kernel kernel, int block,
                             size_t dyn_smem = 0) {
  int dev = 0;
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  cudaFuncAttributes a{};
  CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
  int blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel,
                                                           block, dyn_smem));
  int threads = blocks * block;
  std::printf(
      "  %-14s block=%-4d %3d regs %6zu B smem -> %2d blocks/SM, %4d/%d "
      "threads (%.0f%% occupancy)\n",
      name, block, a.numRegs, a.sharedSizeBytes + dyn_smem, blocks, threads,
      p.maxThreadsPerMultiProcessor,
      100.0 * threads / p.maxThreadsPerMultiProcessor);
}

}  // namespace lab
