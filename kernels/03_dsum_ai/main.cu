// Sum a float vector, accumulating in double -- the point here is the output
// dtype: each block reduces its chunk and writes one fp64 partial, so the
// harness must size the output buffer at 8 bytes/elem and read it back as
// double. Simple-flow only: no float CPU reference to compare a double against.
#include "runner.cuh"

static constexpr int BLOCK = 256;

// Shared-memory tree reduction. Accumulate in double so a long vector doesn't
// shed low bits the way a float accumulator would. One partial per block.
__global__ void dsum_shared(const float* __restrict__ in,
                            double* __restrict__ out, int n) {
  __shared__ double s[BLOCK];
  int t = threadIdx.x;
  int i = blockIdx.x * blockDim.x + t;
  s[t] = (i < n) ? (double)in[i] : 0.0;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (t < stride) s[t] += s[t + stride];
    __syncthreads();
  }
  if (t == 0) out[blockIdx.x] = s[0];
}

// Warp-shuffle reduction: fold within each warp with no shared traffic, then the
// first warp folds the per-warp sums. Same fp64 accumulation, fewer barriers.
__global__ void dsum_warp(const float* __restrict__ in, double* __restrict__ out,
                          int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double v = (i < n) ? (double)in[i] : 0.0;
  for (int off = warpSize / 2; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffff, v, off);
  __shared__ double warp_sums[BLOCK / 32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) warp_sums[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = (lane < blockDim.x / 32) ? warp_sums[lane] : 0.0;
    for (int off = warpSize / 2; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffff, v, off);
    if (lane == 0) out[blockIdx.x] = v;
  }
}

int main(int argc, char** argv) {
  lab::Spec spec("dsum", /*inputs=*/1, /*bytes/elem=*/sizeof(float));
  spec.out_dtype = lab::Dtype::F64;  // 8-byte output elements
  spec.out_numel = [](lab::Shape s) {
    return (long long)lab::blocks(s.w, BLOCK);  // one partial per block
  };
  // Small sizes land <=8 partials (printed as a value list); large ones spill
  // into the shape+mean path. Both variants are deterministic and should agree.
  spec.simple_shapes = {256, 1024, 4096, 65536, 1 << 20};
  // a.out_as<double>() checks the launcher's type against spec.out_dtype above,
  // so a forgotten out_dtype aborts loudly instead of writing past a float-sized
  // buffer. Prefer it to a raw reinterpret_cast for any non-float output.
  spec.variant("shared", dsum_shared, BLOCK, [](lab::Args a) {
    dsum_shared<<<lab::blocks(a.shape.w, BLOCK), BLOCK>>>(
        a.in[0], a.out_as<double>(), a.shape.w);
  });
  spec.variant("warp", dsum_warp, BLOCK, [](lab::Args a) {
    dsum_warp<<<lab::blocks(a.shape.w, BLOCK), BLOCK>>>(
        a.in[0], a.out_as<double>(), a.shape.w);
  });
  return lab::run(argc, argv, spec);
}
