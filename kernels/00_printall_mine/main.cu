// Read the launch geometry back off the device: what grid a given size maps to
// at a fixed block, and each thread's own block/thread index. Simple-flow only:
// no CPU reference, no benchmark. Run it with `simple`.
#include "runner.cuh"

static constexpr int BLOCK = 128;

__global__ void printall(const float* __restrict__ in,
                         float (*__restrict__ out)[4]) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i][0] = gridDim.x;
  out[i][1] = blockDim.x;
  out[i][2] = blockIdx.x;
  out[i][3] = threadIdx.x;
}

int main(int argc, char** argv) {
  lab::Spec spec("printall", /*inputs=*/0, /*bytes/elem=*/sizeof(float));
  spec.out_numel = [](lab::Shape s) {
    return (long long)lab::blocks(s.w, BLOCK) * BLOCK * 4;
  };
  spec.out_row = 4;  // simple mode dumps rows of [gridDim, blockDim, blockIdx, threadIdx]
  spec.simple_shapes = {256, 1024, 4096, 16384, 65536};
  spec.variant("geom", printall, BLOCK, [](lab::Args a) {
    printall<<<lab::blocks(a.shape.w, BLOCK), BLOCK>>>(
        a.in[0], reinterpret_cast<float (*)[4]>(a.out));
  });
  return lab::run(argc, argv, spec);
}
