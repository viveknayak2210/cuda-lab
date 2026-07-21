// 01_vecadd -- the template every other kernel in this repo copies.
//
// Pattern: CPU reference -> awkward-size correctness sweep -> benchmark.
// Run modes:  ./vecadd test    (correctness only, fast, sanitizer-friendly)
//             ./vecadd bench   (timing + roofline, no correctness)
//             ./vecadd profile (single large launch, for nsys/ncu)

#include "../../common/harness.cuh"
#include <cstring>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// CPU reference. Write this FIRST, locally, before you rent anything.
// ---------------------------------------------------------------------------
void vecadd_cpu(const float* a, const float* b, float* c, int n) {
  for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// Variant A: one element per thread. Note the guard -- this is what the
// awkward-size battery exists to catch.
// ---------------------------------------------------------------------------
__global__ void vecadd_naive(const float* __restrict__ a,
                             const float* __restrict__ b, float* __restrict__ c,
                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// Variant B: grid-stride loop. Decouples grid size from problem size, so one
// launch config works for every n and you can tune occupancy independently.
// ---------------------------------------------------------------------------
__global__ void vecadd_gridstride(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c, int n) {
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
struct Buffers {
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  std::vector<float> ha, hb, hc, href;

  void alloc(int n) {
    ha.resize(n); hb.resize(n); hc.resize(n); href.resize(n);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int i = 0; i < n; ++i) { ha[i] = dist(rng); hb[i] = dist(rng); }
    // cudaMalloc(0) is legal and returns nullptr; keep it explicit.
    if (n > 0) {
      CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dc, n * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(da, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(db, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
  }
  void free_all() {
    if (da) CUDA_CHECK(cudaFree(da));
    if (db) CUDA_CHECK(cudaFree(db));
    if (dc) CUDA_CHECK(cudaFree(dc));
    da = db = dc = nullptr;
  }
};

static int run_tests() {
  int failures = 0;
  for (int n : lab::awkward_sizes()) {
    Buffers buf;
    buf.alloc(n);
    vecadd_cpu(buf.ha.data(), buf.hb.data(), buf.href.data(), n);

    const int block = 256;
    const int grid = std::max(1, ceil_div(n, block));

    // -- variant A
    vecadd_naive<<<grid, block>>>(buf.da, buf.db, buf.dc, n);
    KERNEL_CHECK();
    if (n > 0)
      CUDA_CHECK(cudaMemcpy(buf.hc.data(), buf.dc, n * sizeof(float), cudaMemcpyDeviceToHost));
    if (!lab::compare(buf.hc.data(), buf.href.data(), n)) {
      std::fprintf(stderr, "FAIL naive n=%d\n", n); ++failures;
    }

    // -- variant B (deliberately a different, fixed grid to prove it is safe)
    std::fill(buf.hc.begin(), buf.hc.end(), 0.f);
    vecadd_gridstride<<<128, block>>>(buf.da, buf.db, buf.dc, n);
    KERNEL_CHECK();
    if (n > 0)
      CUDA_CHECK(cudaMemcpy(buf.hc.data(), buf.dc, n * sizeof(float), cudaMemcpyDeviceToHost));
    if (!lab::compare(buf.hc.data(), buf.href.data(), n)) {
      std::fprintf(stderr, "FAIL gridstride n=%d\n", n); ++failures;
    }
    buf.free_all();
  }
  if (failures) std::printf("TESTS FAILED (%d)\n", failures);
  else std::printf("all sizes passed\n");
  return failures;
}

static void run_bench() {
  lab::print_device_banner();
  // ptxas -v gave you regs/thread at build time; this is what they buy you.
  lab::report_occupancy("naive", vecadd_naive, 256);
  lab::report_occupancy("gridstride", vecadd_gridstride, 256);
  for (int n : {1 << 20, 1 << 24, 1 << 26}) {
    Buffers buf;
    buf.alloc(n);
    const int block = 256;
    const double bytes = 3.0 * n * sizeof(float);  // 2 read + 1 write

    float ms = lab::time_kernel_ms([&] {
      vecadd_naive<<<ceil_div(n, block), block>>>(buf.da, buf.db, buf.dc, n);
    });
    lab::record("vecadd", "naive", n, ms, bytes);

    ms = lab::time_kernel_ms([&] {
      vecadd_gridstride<<<128, block>>>(buf.da, buf.db, buf.dc, n);
    });
    lab::record("vecadd", "gridstride", n, ms, bytes);

    buf.free_all();
  }
}

static void run_profile() {
  const int n = 1 << 26, block = 256;
  Buffers buf;
  buf.alloc(n);
  vecadd_naive<<<ceil_div(n, block), block>>>(buf.da, buf.db, buf.dc, n);
  KERNEL_CHECK();
  buf.free_all();
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "test";
  if (!std::strcmp(mode, "test")) return run_tests();
  if (!std::strcmp(mode, "bench")) { run_bench(); return 0; }
  if (!std::strcmp(mode, "profile")) { run_profile(); return 0; }
  std::fprintf(stderr, "usage: %s [test|bench|profile]\n", argv[0]);
  return 2;
}
