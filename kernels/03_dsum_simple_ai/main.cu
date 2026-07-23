#include "runner.cuh"

static constexpr int BLOCK = 256;

// Rung 1 -- One atomicAdd per element straight to the global total.
__global__ void reduce_atomic(const float* __restrict__ in,
                              float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out, in[i]);
}

// Rung 2 -- reduce in shared memory first, then ONE atomic per block.
__global__ void reduce_block(const float* __restrict__ in,
                             float* __restrict__ out, int n) {
  __shared__ float s[BLOCK];
  int t = threadIdx.x;
  int i = blockIdx.x * blockDim.x + t;
  s[t] = (i < n) ? in[i] : 0.f;  // out-of-range threads contribute 0
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (t < stride) s[t] += s[t + stride];
    __syncthreads();  // every step must complete before the next reads s[]
  }
  if (t == 0) atomicAdd(out, s[0]);  // one partial per block -> global total
}

// Rung 3 -- coarsen: each thread sums many elements into a register *before* the
// block tree.
__global__ void reduce_coarsened(const float* __restrict__ in,
                                 float* __restrict__ out, int n) {
  __shared__ float s[BLOCK];
  int t = threadIdx.x;
  float sum = 0.f;
  for (int i = blockIdx.x * blockDim.x + t; i < n; i += gridDim.x * blockDim.x)
    sum += in[i];
  s[t] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (t < stride) s[t] += s[t + stride];
    __syncthreads();
  }
  if (t == 0) atomicAdd(out, s[0]);
}

int main(int argc, char** argv) {
  lab::Spec spec("dsum_simple", /*inputs=*/1, /*bytes/elem=*/sizeof(float));
  spec.out_numel = [](lab::Shape) { return 1LL; };  // single float accumulator
  spec.simple_shapes = {256, 1024, 4096, 65536, 1 << 20};

  // Rungs 1 and 2 launch one thread per element (grid scales with n).
  spec.variant("atomic", reduce_atomic, BLOCK, [](lab::Args a) {
    reduce_atomic<<<lab::blocks(a.shape.w, BLOCK), BLOCK>>>(
        a.in[0], a.out, a.shape.w);
  });
  spec.variant("block", reduce_block, BLOCK, [](lab::Args a) {
    reduce_block<<<lab::blocks(a.shape.w, BLOCK), BLOCK>>>(
        a.in[0], a.out, a.shape.w);
  });
  // Rung 3 caps the grid so the coarsening loop actually runs: with a small fixed
  // grid, each thread walks several elements for large n (and just one when
  // blocks(n) is already smaller than the cap).
  spec.variant("coarsened", reduce_coarsened, BLOCK, [](lab::Args a) {
    int grid = std::min(lab::blocks(a.shape.w, BLOCK), 1024);
    reduce_coarsened<<<grid, BLOCK>>>(a.in[0], a.out, a.shape.w);
  });

  return lab::run(argc, argv, spec);
}
