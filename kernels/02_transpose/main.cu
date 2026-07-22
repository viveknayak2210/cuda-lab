// 02_transpose -- the coalescing lesson. Same roofline framework as 01_vecadd
// (still memory-bound: read every element once, write it once), but WHERE the
// bytes land in DRAM is the whole story. The naive kernel reads coalesced and
// writes strided; fixing the write with a shared-memory tile is what separates
// ~a third of peak from ~peak. This is the kernel where nsys/ncu earn their keep.
//
// Pattern: CPU reference -> awkward-size correctness sweep -> benchmark.
// Run modes:  ./02_transpose test    (correctness only, fast, sanitizer-friendly)
//             ./02_transpose bench   (timing + roofline, no correctness)
//             ./02_transpose profile (single large launch, for nsys/ncu)

#include "../../common/harness.cuh"
#include <cstring>
#include <random>
#include <utility>
#include <vector>

// A 32-wide tile matches the 32-bank shared memory / 32-thread warp. BLOCK_ROWS
// < TILE_DIM so each thread walks TILE_DIM/BLOCK_ROWS rows: fewer resident
// threads, same tile, better occupancy than a full 32x32 (1024-thread) block.
static constexpr int TILE_DIM = 32;
static constexpr int BLOCK_ROWS = 8;

// ---------------------------------------------------------------------------
// CPU reference. Input is row-major HxW; output is the row-major WxH transpose,
// so out(x,y) = in(y,x)  ->  out[x*height + y] = in[y*width + x].
// ---------------------------------------------------------------------------
void transpose_cpu(const float* in, float* out, int width, int height) {
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      out[x * height + y] = in[y * width + x];
}

// ---------------------------------------------------------------------------
// Variant A: naive. Reads are coalesced (consecutive threadIdx.x -> consecutive
// input columns), but the write odata[x*height + ...] strides by `height`, so
// each warp scatters 32 separate cache lines. That uncoalesced write is the
// bottleneck the whole file exists to fix. Guards make it awkward-size safe.
// ---------------------------------------------------------------------------
__global__ void transpose_naive(float* __restrict__ odata,
                                const float* __restrict__ idata, int width,
                                int height) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    if (x < width && (y + j) < height)
      odata[x * height + (y + j)] = idata[(y + j) * width + x];
}

// ---------------------------------------------------------------------------
// Variant B: shared-memory tiled. Stage the tile in smem with a coalesced read,
// then write it back transposed -- also coalesced, because the block's OUTPUT
// column is threadIdx.x again. The transpose now happens inside fast smem
// instead of via a strided DRAM write. One flaw remains: reading the tile
// column-wise (tile[threadIdx.x][...]) hits one 32-way bank conflict per access.
// ---------------------------------------------------------------------------
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

  // Swap the block indices: the tile at input block (bx,by) lands at output
  // block (by,bx). Output is WxH, so its row stride is `height`.
  x = blockIdx.y * TILE_DIM + threadIdx.x;  // output column, in [0, height)
  y = blockIdx.x * TILE_DIM + threadIdx.y;  // output row,    in [0, width)
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    if (x < height && (y + j) < width)
      odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
}

// ---------------------------------------------------------------------------
// Variant C: tiled, padded. Pad the smem tile to [TILE_DIM][TILE_DIM+1] so
// column j no longer lands entirely in bank j -- successive rows shift by one
// bank, so the column read spreads across all 32 banks and the conflict is
// gone. One float/row of "wasted" smem buys back the last chunk of bandwidth.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
struct Buffers {
  float *din = nullptr, *dout = nullptr;
  std::vector<float> hin, hout, href;
  int w = 0, h = 0;

  void alloc(int width, int height) {
    w = width;
    h = height;
    const size_t n = (size_t)width * height;
    hin.resize(n); hout.resize(n); href.resize(n);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (size_t i = 0; i < n; ++i) hin[i] = dist(rng);
    // cudaMalloc(0) is legal and returns nullptr; keep the empty case explicit.
    if (n > 0) {
      CUDA_CHECK(cudaMalloc(&din, n * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dout, n * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(din, hin.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
  }
  void free_all() {
    if (din) CUDA_CHECK(cudaFree(din));
    if (dout) CUDA_CHECK(cudaFree(dout));
    din = dout = nullptr;
  }
};

// One block per TILE_DIM x TILE_DIM output tile. max(1,...) keeps a 0-sized dim
// from producing an illegal 0-block launch -- the in-kernel guards do the rest.
static dim3 grid_for(int width, int height) {
  return dim3(std::max(1, ceil_div(width, TILE_DIM)),
              std::max(1, ceil_div(height, TILE_DIM)));
}

// The 2D analog of lab::awkward_sizes(): {width, height} pairs that straddle
// tile and warp boundaries, go thin/degenerate, and end on non-square,
// non-power-of-two dims (incl. a prime side) that a naive index would fumble.
static const std::vector<std::pair<int, int>>& awkward_shapes() {
  static const std::vector<std::pair<int, int>> v = {
      {0, 0},   {1, 1},    {1, 32},   {32, 1},   {3, 5},    {31, 31},
      {32, 32}, {33, 33},  {32, 33},  {33, 32},  {64, 64},  {100, 37},
      {129, 257} /* non-multiples of 32; 257 prime */, {200, 200},
      {512, 500}};
  return v;
}

static int run_tests() {
  int failures = 0;
  for (const auto& shape : awkward_shapes()) {
    const int w = shape.first, h = shape.second;
    Buffers buf;
    buf.alloc(w, h);
    transpose_cpu(buf.hin.data(), buf.href.data(), w, h);
    const size_t n = (size_t)w * h;

    const dim3 block(TILE_DIM, BLOCK_ROWS);
    const dim3 grid = grid_for(w, h);

    // Launch a variant into a freshly-zeroed output, copy back, compare. A
    // lambda keeps this to one place; the launches themselves stay explicit
    // (by name, not through a pointer) so nvcc emits a normal launch stub.
    auto verify = [&](const char* name) {
      if (n > 0)
        CUDA_CHECK(cudaMemcpy(buf.hout.data(), buf.dout, n * sizeof(float),
                              cudaMemcpyDeviceToHost));
      if (!lab::compare(buf.hout.data(), buf.href.data(), (long long)n)) {
        std::fprintf(stderr, "FAIL %s %dx%d\n", name, w, h);
        ++failures;
      }
    };

    if (n > 0) CUDA_CHECK(cudaMemset(buf.dout, 0, n * sizeof(float)));
    transpose_naive<<<grid, block>>>(buf.dout, buf.din, w, h);
    KERNEL_CHECK();
    verify("naive");

    if (n > 0) CUDA_CHECK(cudaMemset(buf.dout, 0, n * sizeof(float)));
    transpose_tiled<<<grid, block>>>(buf.dout, buf.din, w, h);
    KERNEL_CHECK();
    verify("tiled");

    if (n > 0) CUDA_CHECK(cudaMemset(buf.dout, 0, n * sizeof(float)));
    transpose_tiled_pad<<<grid, block>>>(buf.dout, buf.din, w, h);
    KERNEL_CHECK();
    verify("tiled_pad");

    buf.free_all();
  }
  if (failures) std::printf("TESTS FAILED (%d)\n", failures);
  else std::printf("all sizes passed\n");
  // Exit codes are mod 256 -- a raw count of exactly 256 would read as success.
  return failures ? 1 : 0;
}

static void run_bench() {
  lab::print_device_banner();
  // ptxas -v gave you regs/thread at build time; this is what they buy you.
  // The tiled variants also report their smem/block, so you see the occupancy
  // cost of the staging tile directly next to the bandwidth it wins back.
  lab::report_occupancy("naive", transpose_naive, TILE_DIM * BLOCK_ROWS);
  lab::report_occupancy("tiled", transpose_tiled, TILE_DIM * BLOCK_ROWS);
  lab::report_occupancy("tiled_pad", transpose_tiled_pad, TILE_DIM * BLOCK_ROWS);
  // Square matrices whose element counts match 01_vecadd's sweep (2^20..2^26),
  // so the two kernels' %-of-peak are read on the same axis.
  for (int side : {1024, 4096, 8192}) {
    const int w = side, h = side;
    Buffers buf;
    buf.alloc(w, h);
    const long long n = (long long)w * h;
    const double bytes = 2.0 * n * sizeof(float);  // 1 read + 1 write
    const dim3 block(TILE_DIM, BLOCK_ROWS);
    const dim3 grid = grid_for(w, h);

    float ms = lab::time_kernel_ms(
        [&] { transpose_naive<<<grid, block>>>(buf.dout, buf.din, w, h); });
    lab::record("transpose", "naive", n, ms, bytes);

    ms = lab::time_kernel_ms(
        [&] { transpose_tiled<<<grid, block>>>(buf.dout, buf.din, w, h); });
    lab::record("transpose", "tiled", n, ms, bytes);

    ms = lab::time_kernel_ms(
        [&] { transpose_tiled_pad<<<grid, block>>>(buf.dout, buf.din, w, h); });
    lab::record("transpose", "tiled_pad", n, ms, bytes);

    buf.free_all();
  }
}

static void run_profile() {
  // Profile the naive variant on purpose: it's the one whose uncoalesced write
  // you want to see in the timeline / counters.
  const int w = 8192, h = 8192;
  Buffers buf;
  buf.alloc(w, h);
  const dim3 block(TILE_DIM, BLOCK_ROWS);
  transpose_naive<<<grid_for(w, h), block>>>(buf.dout, buf.din, w, h);
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
