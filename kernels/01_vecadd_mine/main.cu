#include "runner.cuh"

static constexpr int BLOCK = 256;

__global__ void vecadd_contiguous_per_thread(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n) {
      int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
      int total_threads = gridDim.x * blockDim.x;
      int chunk_size = (n + total_threads - 1) / total_threads;
      int start_point = global_thread_id * chunk_size;
      int end_point = min(start_point + chunk_size, n);
      for (int i = start_point; i<end_point; i++){
        c[i] = a[i] + b[i];
      }
}

static void vecadd_cpu(const float* const* in, float* out, lab::Shape s) {
  for (int i = 0; i < s.w; ++i) out[i] = in[0][i] + in[1][i];
}

int main(int argc, char** argv) {
  lab::Spec spec("vecadd", /*inputs=*/2, /*bytes/elem=*/3 * sizeof(float));
  spec.reference = vecadd_cpu;
  spec.test_shapes = lab::awkward_1d();
  spec.bench_shapes = {1 << 20, 1 << 24, 1 << 26};

  spec.variant("contig_per_thread", vecadd_contiguous_per_thread, BLOCK,
               [](lab::Args a) {
                 vecadd_contiguous_per_thread<<<128, BLOCK>>>(
                     a.in[0], a.in[1], a.out, a.shape.w);
               });

  return lab::run(argc, argv, spec);
}