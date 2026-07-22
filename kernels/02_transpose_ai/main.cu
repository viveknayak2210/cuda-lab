// 02_transpose_ai -- coalescing. Same 8-bytes-per-element roofline as a plain copy;
// the entire story is whether a warp's writes land in one cache line or 32.
// Modes: test | bench | profile  (see common/runner.cuh).
#include "runner.cuh"

// 32 wide matches the warp and the 32 shared-memory banks. BLOCK_ROWS < TILE_DIM
// so each thread walks TILE_DIM/BLOCK_ROWS rows: same tile, fewer resident
// threads, better occupancy than a 32x32 (1024-thread) block.
static constexpr int TILE_DIM = 32;
static constexpr int BLOCK_ROWS = 8;
static constexpr dim3 BLOCK(TILE_DIM, BLOCK_ROWS);

// One block per TILE_DIM x TILE_DIM output tile.
static dim3 grid(lab::Shape s) {
  return dim3(lab::blocks(s.w, TILE_DIM), lab::blocks(s.h, TILE_DIM));
}

// out(x,y) = in(y,x): row-major HxW in, row-major WxH out.
static void transpose_cpu(const float* const* in, float* out, lab::Shape s) {
  for (int y = 0; y < s.h; ++y)
    for (int x = 0; x < s.w; ++x) out[x * s.h + y] = in[0][y * s.w + x];
}

// Reads coalesce (consecutive threadIdx.x -> consecutive columns), but the write
// strides by `height`, scattering each warp across 32 cache lines. That write is
// the bottleneck the rest of this file exists to remove.
__global__ void transpose_naive(float* __restrict__ odata,
                                const float* __restrict__ idata, int width,
                                int height) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    if (x < width && (y + j) < height)
      odata[x * height + (y + j)] = idata[(y + j) * width + x];
}

// Stage the tile in shared memory, then write it back transposed: the output
// column is threadIdx.x again, so both DRAM accesses coalesce and the transpose
// happens in smem. Reading the tile column-wise still costs one 32-way bank
// conflict per access.
__global__ void transpose_tiled(float* __restrict__ odata,
                                const float* __restrict__ idata, int width,
                                int height) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    if (x < width && (y + j) < height)
      tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

  __syncthreads();

  // Swapped block indices: input block (bx,by) lands at output block (by,bx),
  // and the WxH output has row stride `height`.
  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    if (x < height && (y + j) < width)
      odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
}

// One float of padding per row shifts each row by a bank, so a column read
// spreads over all 32 banks instead of hitting one. Same code otherwise.
__global__ void transpose_tiled_pad(float* __restrict__ odata,
                                    const float* __restrict__ idata, int width,
                                    int height) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    if (x < width && (y + j) < height)
      tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    if (x < height && (y + j) < width)
      odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main(int argc, char** argv) {
  lab::Spec spec("transpose", /*inputs=*/1, /*bytes/elem=*/2 * sizeof(float));
  spec.reference = transpose_cpu;
  spec.test_shapes = lab::awkward_2d();
  // Element counts match 01_vecadd_ai's sweep (2^20..2^26), so both kernels'
  // %-of-peak are read on the same axis.
  spec.bench_shapes = {{1024, 1024}, {4096, 4096}, {8192, 8192}};

  spec.variant("naive", transpose_naive, BLOCK, [](lab::Args a) {
    transpose_naive<<<grid(a.shape), BLOCK>>>(a.out, a.in[0], a.shape.w,
                                              a.shape.h);
  });
  spec.variant("tiled", transpose_tiled, BLOCK, [](lab::Args a) {
    transpose_tiled<<<grid(a.shape), BLOCK>>>(a.out, a.in[0], a.shape.w,
                                              a.shape.h);
  });
  spec.variant("tiled_pad", transpose_tiled_pad, BLOCK, [](lab::Args a) {
    transpose_tiled_pad<<<grid(a.shape), BLOCK>>>(a.out, a.in[0], a.shape.w,
                                                  a.shape.h);
  });

  return lab::run(argc, argv, spec);
}
