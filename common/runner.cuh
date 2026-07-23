#pragma once
#include "harness.cuh"
#include <algorithm>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

// The driver behind every kernels/NN_*/main.cu. A kernel file supplies only the
// lesson -- a CPU reference, the __global__ variants, and a Spec wiring them up.

// Buffer model: `inputs` device inputs + one output, each Shape::numel() floats
// (set Spec::out_numel when the output is a different size, e.g. a reduction).
// Kernels whose inputs differ in size (sgemm) need this widened.

namespace lab {

constexpr int kMaxInputs = 4;

// 1-D kernels use `w` and leave `h == 1`.
struct Shape {
  int w = 0, h = 1;
  constexpr Shape() = default;
  constexpr Shape(int w_, int h_ = 1) : w(w_), h(h_) {}
  long long numel() const { return (long long)w * h; }
  std::string label() const {
    return h == 1 ? "n=" + std::to_string(w)
                  : std::to_string(w) + "x" + std::to_string(h);
  }
};

// Grid geometry. Never zero blocks: an empty shape still has to produce a legal
// launch, and the kernel's own bounds guard does the rest.
inline int blocks(int items, int per_block) {
  return std::max(1, ceil_div(items, per_block));
}

// What a variant's launcher gets handed: device pointers + the shape under test.
struct Args {
  float* out = nullptr;
  const float* in[kMaxInputs] = {};
  Shape shape;
};

struct Variant {
  std::string name;
  std::function<void(Args)> launch;
  std::function<void()> occupancy;
};

// ---------------------------------------------------------------------------
// The awkward-size batteries. Every kernel runs against all of these before it
// is allowed to be called correct: empty, single, sub-warp, exact warp/block,
// +-1 on both, ragged multi-block, and a large prime.
// ---------------------------------------------------------------------------
inline const std::vector<Shape>& awkward_1d() {
  static const std::vector<Shape> v = {
      0,    1,     2,     31,    32,     33,     63,     64,     65,
      127,  128,   129,   255,   256,    257,    511,    512,    513,
      1023, 1024,  1025,  4095,  4096,   4097,   100000, 1048573 /* prime */,
      1 << 22};
  return v;
}

// The 2-D analog: straddles tile and warp boundaries, goes thin/degenerate, and
// ends on non-square, non-power-of-two dims (257 prime).
inline const std::vector<Shape>& awkward_2d() {
  static const std::vector<Shape> v = {
      {0, 0},   {1, 1},    {1, 32},   {32, 1},   {3, 5},
      {31, 31}, {32, 32},  {33, 33},  {32, 33},  {33, 32},
      {64, 64}, {100, 37}, {129, 257} /* 257 prime */,
      {200, 200}, {512, 500}};
  return v;
}

// ---------------------------------------------------------------------------
struct Spec {
  std::string name;         // the `kernel` column in bench.csv
  int inputs;
  double bytes_per_elem;    // DRAM traffic per element -- the roofline divisor
  std::function<void(const float* const* in, float* out, Shape)> reference;
  std::function<long long(Shape)> out_numel;  // default: Shape::numel()
  int out_row = 0;  // >0: simple mode also dumps the first/last rows of this width
  std::vector<Shape> test_shapes, bench_shapes;
  std::vector<Shape> simple_shapes;  // `simple` mode; empty -> sampled from test_shapes
  std::vector<Variant> variants;

  Spec(std::string kernel, int n_inputs, double bytes_per_elem)
      : name(std::move(kernel)), inputs(n_inputs),
        bytes_per_elem(bytes_per_elem) {
    if (inputs > kMaxInputs) {
      std::fprintf(stderr, "too many inputs (max %d)\n", kMaxInputs);
      std::exit(1);
    }
  }

  // `kernel` is only read for the occupancy report -- cudaFuncGetAttributes
  // needs the symbol, and only the <<<>>> inside `launch` knows the arguments.
  template <typename Kernel>
  void variant(std::string vname, Kernel kernel, dim3 block,
               std::function<void(Args)> launch) {
    const int threads = block.x * block.y * block.z;
    auto occupancy = [vname, kernel, threads] {
      report_occupancy(vname.c_str(), kernel, threads);
    };
    variants.push_back(
        {std::move(vname), std::move(launch), std::move(occupancy)});
  }

  long long out_elems(Shape s) const {
    return out_numel ? out_numel(s) : s.numel();
  }
};

// ---------------------------------------------------------------------------
// Host + device buffers for one shape. Inputs are filled from a fixed seed, so
// a failure is reproducible; the output is zeroed before every launch, so a
// kernel that skips elements fails the compare instead of reading stale data.
// ---------------------------------------------------------------------------
class Buffers {
 public:
  Buffers(const Spec& spec, Shape shape)
      : shape_(shape), n_(shape.numel()), out_n_(spec.out_elems(shape)) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    h_in_.resize(spec.inputs);
    for (auto& host : h_in_) {
      host.resize(n_);
      for (long long i = 0; i < n_; ++i) host[i] = dist(rng);
    }
    for (const auto& host : h_in_) h_in_ptr_.push_back(host.data());
    h_out_.resize(out_n_);
    h_ref_.resize(out_n_);

    // cudaMalloc(0) is legal and hands back nullptr; keep the empty case explicit.
    d_in_.assign(spec.inputs, nullptr);
    if (n_ > 0) {
      for (int k = 0; k < spec.inputs; ++k) {
        CUDA_CHECK(cudaMalloc(&d_in_[k], n_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in_[k], h_in_[k].data(), n_ * sizeof(float),
                              cudaMemcpyHostToDevice));
      }
    }
    if (out_n_ > 0) CUDA_CHECK(cudaMalloc(&d_out_, out_n_ * sizeof(float)));
  }

  ~Buffers() {
    for (float* p : d_in_)
      if (p) CUDA_CHECK(cudaFree(p));
    if (d_out_) CUDA_CHECK(cudaFree(d_out_));
  }
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;

  Args args() const {
    Args a;
    a.out = d_out_;
    for (size_t k = 0; k < d_in_.size(); ++k) a.in[k] = d_in_[k];
    a.shape = shape_;
    return a;
  }

  void clear_out() {
    if (out_n_ > 0) CUDA_CHECK(cudaMemset(d_out_, 0, out_n_ * sizeof(float)));
  }
  void fetch() {
    if (out_n_ > 0)
      CUDA_CHECK(cudaMemcpy(h_out_.data(), d_out_, out_n_ * sizeof(float),
                            cudaMemcpyDeviceToHost));
  }

  const float* const* host_in() const { return h_in_ptr_.data(); }
  float* host_ref() { return h_ref_.data(); }
  const float* host_out() const { return h_out_.data(); }
  long long out_n() const { return out_n_; }

 private:
  Shape shape_;
  long long n_, out_n_;
  std::vector<std::vector<float>> h_in_;
  std::vector<const float*> h_in_ptr_;
  std::vector<float> h_out_, h_ref_;
  std::vector<float*> d_in_;
  float* d_out_ = nullptr;
};

// ---------------------------------------------------------------------------
// Run modes.
// ---------------------------------------------------------------------------
inline int run_tests(const Spec& spec) {
  int failures = 0;
  for (Shape s : spec.test_shapes) {
    Buffers buf(spec, s);
    spec.reference(buf.host_in(), buf.host_ref(), s);
    for (const Variant& v : spec.variants) {
      buf.clear_out();
      v.launch(buf.args());
      KERNEL_CHECK();
      buf.fetch();
      if (!compare(buf.host_out(), buf.host_ref(), buf.out_n())) {
        std::fprintf(stderr, "FAIL %s %s\n", v.name.c_str(), s.label().c_str());
        ++failures;
      }
    }
  }
  if (failures)
    std::printf("TESTS FAILED (%d)\n", failures);
  else
    std::printf("all %zu shapes x %zu variants passed\n",
                spec.test_shapes.size(), spec.variants.size());
  // Exit codes are mod 256 -- a raw count of exactly 256 would read as success.
  return failures ? 1 : 0;
}

inline void run_bench(const Spec& spec) {
  print_device_banner();
  // ptxas -v gives regs/thread and smem/block at build time
  for (const Variant& v : spec.variants) v.occupancy();
  for (Shape s : spec.bench_shapes) {
    Buffers buf(spec, s);
    const Args a = buf.args();
    for (const Variant& v : spec.variants) {
      float ms = time_kernel_ms([&] { v.launch(a); });
      record(spec.name, v.name, s.numel(), ms,
             spec.bytes_per_elem * (double)s.numel());
    }
  }
}

// The ~5 sizes `simple` mode sweeps when a kernel doesn't set simple_shapes: an
// increasing ramp sampled from test_shapes, so it inherits the kernel's own 1-D
// or 2-D shape and drops the empty case. Deterministic -- no timing, no compare.
inline std::vector<Shape> simple_ramp(const std::vector<Shape>& from, int k = 5) {
  std::vector<Shape> v;
  for (Shape s : from)
    if (s.numel() > 0) v.push_back(s);
  std::sort(v.begin(), v.end(),
            [](Shape a, Shape b) { return a.numel() < b.numel(); });
  v.erase(std::unique(v.begin(), v.end(),
                      [](Shape a, Shape b) { return a.numel() == b.numel(); }),
          v.end());
  if ((int)v.size() <= k) return v;
  std::vector<Shape> out;
  for (int i = 0; i < k; ++i)
    out.push_back(v[(long long)i * (v.size() - 1) / (k - 1)]);
  return out;
}

// `simple` mode: launch one variant across a handful of sizes and print a quick
// signature of what came back -- shape + mean for a tensor, the raw values for a
// scalar/tiny output. No CPU compare, no timing; the fast "did it produce
// something sane" loop, distinct from the `test` correctness gate.
inline int run_simple(const Spec& spec, const char* want) {
  const Variant* v = &spec.variants.front();
  if (want) {
    v = nullptr;
    for (const Variant& c : spec.variants)
      if (c.name == want) v = &c;
    if (!v) {
      std::fprintf(stderr, "unknown variant '%s'\n", want);
      return 2;
    }
  }
  std::vector<Shape> shapes =
      spec.simple_shapes.empty() ? simple_ramp(spec.test_shapes)
                                 : spec.simple_shapes;
  std::printf("simple: %s / %s -- %zu sizes (unverified)\n", spec.name.c_str(),
              v->name.c_str(), shapes.size());
  for (Shape s : shapes) {
    Buffers buf(spec, s);
    buf.clear_out();
    v->launch(buf.args());
    KERNEL_CHECK();
    buf.fetch();
    const long long n = buf.out_n();
    const float* o = buf.host_out();
    if (n <= 8) {  // scalar / tiny output -> the values themselves
      std::printf("  in=%-10s out numel=%lld  vals=[", s.label().c_str(), n);
      for (long long i = 0; i < n; ++i)
        std::printf("%s%.6g", i ? ", " : "", o[i]);
      std::printf("]\n");
    } else {  // tensor -> shape + mean, plus an optional first/last-rows dump
      double sum = 0.0;
      for (long long i = 0; i < n; ++i) sum += o[i];
      std::printf("  in=%-10s out numel=%lld  mean=%.6g\n", s.label().c_str(),
                  n, sum / (double)n);
      // out_row>0 means the flat buffer is [rows][out_row]; show the head and
      // tail so a per-row kernel (e.g. one value per thread) is eyeballable.
      if (spec.out_row > 0) {
        const int w = spec.out_row;
        const long long rows = n / w, show = 3;  // `show` rows at each end
        for (long long r = 0; r < rows; ++r) {
          if (rows > 2 * show && r == show) {
            std::printf("      ... (%lld rows elided)\n", rows - 2 * show);
            r = rows - show - 1;  // jump to the tail
            continue;
          }
          std::printf("      row %-6lld [", r);
          for (int j = 0; j < w; ++j)
            std::printf("%s%g", j ? ", " : "", o[r * w + j]);
          std::printf("]\n");
        }
      }
    }
  }
  return 0;
}

// One launch, largest bench shape
inline int run_profile(const Spec& spec, const char* want) {
  const Variant* v = &spec.variants.front();
  if (want) {
    v = nullptr;
    for (const Variant& c : spec.variants)
      if (c.name == want) v = &c;
    if (!v) {
      std::fprintf(stderr, "unknown variant '%s'\n", want);
      return 2;
    }
  }
  Buffers buf(spec, spec.bench_shapes.back());
  v->launch(buf.args());
  KERNEL_CHECK();
  return 0;
}

inline int run(int argc, char** argv, const Spec& spec) {
  const char* mode = argc > 1 ? argv[1] : "test";

  // Every mode has to have something to launch.
  if (spec.variants.empty()) {
    std::fprintf(stderr, "%s: no variants\n", spec.name.c_str());
    return 2;
  }

  // `simple` is the lightweight path -- no CPU compare, no timing -- so it needs
  // neither a reference nor bench_shapes, only something to size the sweep. That
  // lets a kernel be simple-only, with no test/bench wiring at all.
  if (!std::strcmp(mode, "simple")) {
    if (spec.simple_shapes.empty() && spec.test_shapes.empty()) {
      std::fprintf(stderr, "%s: simple needs simple_shapes or test_shapes\n",
                   spec.name.c_str());
      return 2;
    }
    return run_simple(spec, argc > 2 ? argv[2] : nullptr);
  }

  // test/bench/profile are the full contract: a reference plus both shape lists.
  if (!spec.reference || spec.test_shapes.empty() || spec.bench_shapes.empty()) {
    std::fprintf(stderr,
                 "%s: incomplete Spec (reference/test_shapes/bench_shapes)\n",
                 spec.name.c_str());
    return 2;
  }
  if (!std::strcmp(mode, "test")) return run_tests(spec);
  if (!std::strcmp(mode, "bench")) { run_bench(spec); return 0; }
  if (!std::strcmp(mode, "profile"))
    return run_profile(spec, argc > 2 ? argv[2] : nullptr);
  std::fprintf(stderr,
               "usage: %s [test|bench|profile [variant]|simple [variant]]\n",
               argv[0]);
  return 2;
}

}  // namespace lab
