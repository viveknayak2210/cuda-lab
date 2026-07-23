#include "runner.cuh"

static constexpr int BLOCK = 256;

__global__ void sum_reduction_naive(float const * __restrict__ in, float* __restrict__ out, int n){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid<n){
    atomicAdd(out, in[tid]);
  }
}

static void sum_reduction_cpu(const float* const* in, float* out, lab::Shape s) {
  out[0] = 0.f;
  for (int i = 0; i < s.w; ++i) out[0] += in[0][i];
}


int main(int argc, char** argv) {
  lab::Spec spec("sum_reduction", /*inputs=*/1, /*bytes/elem=*/sizeof(float));
  spec.out_numel = [](lab::Shape) { return 1LL; };  // one accumulator (float, the default)
  spec.reference = sum_reduction_cpu;
  spec.test_shapes = lab::awkward_1d();
  spec.bench_shapes = {1 << 20, 1 << 24, 1 << 26};
  spec.simple_shapes = {256, 1024, 4096, 65536, 1 << 20};
  spec.variant("naive", sum_reduction_naive, BLOCK, [](lab::Args a) {
    sum_reduction_naive<<<lab::blocks(a.shape.w, BLOCK), BLOCK>>>(
        a.in[0], a.out, a.shape.w);
  });
  return lab::run(argc, argv, spec);
}