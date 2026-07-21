#pragma once
#include "check.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

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
inline bool compare(const float* got, const float* want, int n,
                    float rtol = 1e-5f, float atol = 1e-6f,
                    int max_report = 5) {
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    float diff = std::fabs(got[i] - want[i]);
    float tol = atol + rtol * std::fabs(want[i]);
    if (!(diff <= tol) || std::isnan(got[i])) {
      if (bad < max_report) {
        std::fprintf(stderr, "  mismatch @%d: got %.9g want %.9g (|d|=%.3g)\n",
                     i, got[i], want[i], diff);
      }
      ++bad;
    }
  }
  if (bad) std::fprintf(stderr, "  %d/%d elements wrong\n", bad, n);
  return bad == 0;
}

// ---------------------------------------------------------------------------
// Achievable bandwidth probe. Compare against THIS, not the spec-sheet number:
// the spec number is unreachable and makes every kernel look worse than it is.
// Cached to results/peak_bw.txt so it is measured once per pod.
// ---------------------------------------------------------------------------
__global__ void _triad(float* __restrict__ dst, const float* __restrict__ src,
                       size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) dst[i] = src[i] * 2.0f;
}

inline float measure_peak_bw_gbs() {
  if (FILE* f = std::fopen("results/peak_bw.txt", "r")) {
    float cached = 0.f;
    int ok = std::fscanf(f, "%f", &cached);
    std::fclose(f);
    if (ok == 1 && cached > 0.f) return cached;
  }
  const size_t n = 1ull << 26;  // 256 MB in, 256 MB out
  float *a = nullptr, *b = nullptr;
  CUDA_CHECK(cudaMalloc(&a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(a, 0, n * sizeof(float)));

  int dev = 0, sms = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  const int block = 256, grid = sms * 32;

  float ms = time_kernel_ms([&] { _triad<<<grid, block>>>(b, a, n); }, 3, 20);
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));

  float gbs = (2.0f * n * sizeof(float)) / (ms * 1e6f);
  if (FILE* f = std::fopen("results/peak_bw.txt", "w")) {
    std::fprintf(f, "%.3f\n", gbs);
    std::fclose(f);
  }
  return gbs;
}

// ---------------------------------------------------------------------------
// One row of the results CSV. Append-only so runs accumulate across sessions.
// ---------------------------------------------------------------------------
inline void record(const std::string& kernel, const std::string& variant, int n,
                   float ms, double bytes_moved, double flops = 0.0) {
  FILE* f = std::fopen("results/bench.csv", "a");
  if (!f) return;
  std::fseek(f, 0, SEEK_END);
  if (std::ftell(f) == 0)
    std::fprintf(f, "kernel,variant,n,ms,gbs,gflops,pct_peak_bw\n");

  static float peak = measure_peak_bw_gbs();
  double gbs = bytes_moved / (ms * 1e6);
  double gflops = flops / (ms * 1e6);
  std::fprintf(f, "%s,%s,%d,%.6f,%.2f,%.2f,%.1f\n", kernel.c_str(),
               variant.c_str(), n, ms, gbs, gflops, 100.0 * gbs / peak);
  std::fclose(f);

  std::printf("  %-14s n=%-9d %8.4f ms  %7.1f GB/s  %5.1f%% of peak\n",
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

}  // namespace lab
