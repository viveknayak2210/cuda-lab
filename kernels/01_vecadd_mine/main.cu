#include "runner.cuh"

static constexpr int BLOCK = 256;

__global__ void vecadd_per_thread(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n) {
      int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
      if (global_thread_id < n){
        c[global_thread_id] = a[global_thread_id] + b[global_thread_id];
      }
}

__global__ void vecadd_strided(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n) {
      int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
      int stride = gridDim.x * blockDim.x;
      for (int i=global_thread_id; i<n; i+=stride)
        c[i] = a[i] + b[i];
}

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

  spec.variant("per_thread", vecadd_per_thread, BLOCK, [](lab::Args a) {
    vecadd_per_thread<<<lab::blocks(a.shape.w, BLOCK), BLOCK>>>(
        a.in[0], a.in[1], a.out, a.shape.w);
  });
  spec.variant("strided", vecadd_strided, BLOCK, [](lab::Args a) {
    vecadd_strided<<<128, BLOCK>>>(a.in[0], a.in[1], a.out, a.shape.w);
  });
  spec.variant("strided_full", vecadd_strided, BLOCK, [](lab::Args a) {
    int sms = 0, bps = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, vecadd_strided, BLOCK, 0);
    int grid = std::min(lab::blocks(a.shape.w, BLOCK), sms * bps);
    vecadd_strided<<<grid, BLOCK>>>(a.in[0], a.in[1], a.out, a.shape.w);
  });
  spec.variant("contig_per_thread", vecadd_contiguous_per_thread, BLOCK,
               [](lab::Args a) {
                 vecadd_contiguous_per_thread<<<128, BLOCK>>>(
                     a.in[0], a.in[1], a.out, a.shape.w);
               });

  return lab::run(argc, argv, spec);
}