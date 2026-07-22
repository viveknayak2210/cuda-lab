// 01_vecadd -- elementwise add. 12 bytes moved per element and one flop to show
// for it: pure bandwidth, the baseline every other kernel's roofline is read
// against.  Modes: test | bench | profile  (see common/runner.cuh).
#include "runner.cuh"

static constexpr int BLOCK = 256;

// One element per thread. The guard is what the awkward-size battery exists to
// catch.
__global__ void vecadd_naive(const float* __restrict__ a,
                             const float* __restrict__ b, float* __restrict__ c,
                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

// Grid-stride: decouples grid size from n, so one launch config covers every
// size and occupancy can be tuned independently of the problem.
__global__ void vecadd_gridstride(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c, int n) {
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    c[i] = a[i] + b[i];
}

static void vecadd_cpu(const float* const* in, float* out, lab::Shape s) {
  for (int i = 0; i < s.w; ++i) out[i] = in[0][i] + in[1][i];
}

int main(int argc, char** argv) {
  lab::Spec spec("vecadd", /*inputs=*/2, /*bytes/elem=*/3 * sizeof(float));
  spec.reference = vecadd_cpu;
  spec.test_shapes = lab::awkward_1d();
  spec.bench_shapes = {1 << 20, 1 << 24, 1 << 26};

  spec.variant("naive", vecadd_naive, BLOCK, [](lab::Args a) {
    vecadd_naive<<<lab::blocks(a.shape.w, BLOCK), BLOCK>>>(a.in[0], a.in[1],
                                                           a.out, a.shape.w);
  });
  // Deliberately a fixed grid, sized to neither n nor the card: proof that the
  // stride loop is safe for any n.
  spec.variant("gridstride", vecadd_gridstride, BLOCK, [](lab::Args a) {
    vecadd_gridstride<<<128, BLOCK>>>(a.in[0], a.in[1], a.out, a.shape.w);
  });

  return lab::run(argc, argv, spec);
}
