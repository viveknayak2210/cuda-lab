// Sum a float vector, accumulating in double
#include "runner.cuh"

static constexpr int BLOCK = 256;

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
  spec.simple_shapes = {256, 1024, 4096, 65536, 1 << 20};
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
