# CLAUDE.md

Personal CUDA kernel lab. Kernels are written and type-checked on a Mac with no
GPU; a rented RunPod box runs the battery as a **batch job**, never as a
workstation. GPU minutes are the scarce resource and the whole repo is shaped
around spending as few as possible.

## Ground rules

1. **Nothing runs locally.** No GPU, no driver — `cudaMalloc` fails on the Mac.
   `nvcc` is ahead-of-time, so you *can* compile, count registers, and read
   PTX/SASS offline. Never report a kernel as working because it compiled.
2. **Type-check every change** in the offline container before claiming done.
   Anything that only fails on the pod costs real money.
3. **Correctness gates timing.** `test` must pass the awkward-size sweep before
   any benchmark number is worth reporting or committing.
4. **Roofline, not raw ms.** Numbers are read as % of *measured achievable*
   bandwidth (`results/peak_bw.txt`), never % of the spec sheet.
5. `bin/`, `build/`, `results/`, `local/pod.env` are generated/private
   (gitignored). `course_plan.md` is the user's personal curriculum, also
   gitignored — useful context, don't edit it.

## Commands

```bash
# offline toolchain (Docker, arm64 CUDA 12.6 — no GPU needed)
./local/cudash make ARCH=86              # type-check the whole tree
./local/cudash make regs K=01_vecadd     # registers/thread + smem/block
./local/cudash make sass K=01_vecadd ARCH=90a
# ...cudash uses `docker run -it`; from a non-TTY agent shell use:
docker run --rm --platform=linux/arm64 -v "$PWD:/work" -w /work cudalab-offline \
  make regs K=01_vecadd ARCH=86

# paid pod session (from the Mac; needs local/pod.env)
./local/session.sh --boot                # first run on a fresh pod
./local/session.sh [kernel]              # sync -> battery -> pull results
./local/session.sh --stop                # NO auto-stop; this is the off switch
```

`/runpod_setup HOST:PORT` and `/runpod_run [kernel]` (`.claude/commands/`) drive
those from Claude Code. `ncu`/`PROFILE=1` is blocked on RunPod
(`ERR_NVGPUCTRPERM`) — use the nsys timeline instead, it needs no perf counters.

## Layout

```
kernels/NN_name/main.cu   ONE kernel family: CPU reference + __global__ variants + Spec
common/check.cuh          CUDA_CHECK / KERNEL_CHECK / ceil_div
common/harness.cuh        instruments: event timing, compare, peak-BW probe,
                          CSV record (with provenance), occupancy report
common/runner.cuh         the driver: Shape, Args, Variant, Spec, Buffers,
                          awkward-size batteries, lab::run (test|bench|profile)
Makefile                  arch auto-detect, kernel auto-discovery, ptx/sass/regs
scripts/bootstrap.sh      one-time pod setup (identify GPU + smoke test)
scripts/run.sh            the battery: build -> test -> sanitize -> bench -> nsys
local/                    session.sh (Mac-side driver), cudash, Dockerfile.offline
README.md / GUIDE.md      the workflow in prose; GUIDE.md is the long form
                          (phase table, offline nvcc, what to write before a pod)
```

`NVCCFLAGS` carries `-Icommon`, so kernel files include `"runner.cuh"` directly.
The Makefile discovers `kernels/*/main.cu` — a new directory needs no Makefile
edit, and `bin/<dir>` is the binary name `scripts/run.sh` iterates over.

## The kernel-file contract

A `main.cu` holds **only** the lesson: the CPU reference, the `__global__`
variants, and a `Spec` wiring them up. Everything mechanical — allocation, H2D,
zeroing, the size sweep, comparison, event timing, CSV rows, occupancy, mode
dispatch — lives in `common/runner.cuh`.

```cpp
#include "runner.cuh"

__global__ void foo_naive(...) { ... }

static void foo_cpu(const float* const* in, float* out, lab::Shape s) { ... }

int main(int argc, char** argv) {
  lab::Spec spec("foo", /*inputs=*/1, /*bytes/elem=*/2 * sizeof(float));
  spec.reference = foo_cpu;
  spec.test_shapes = lab::awkward_2d();          // or lab::awkward_1d()
  spec.bench_shapes = {{1024, 1024}, {4096, 4096}};
  spec.variant("naive", foo_naive, BLOCK, [](lab::Args a) {
    foo_naive<<<grid(a.shape), BLOCK>>>(a.out, a.in[0], a.shape.w, a.shape.h);
  });
  return lab::run(argc, argv, spec);
}
```

- `Spec("name", inputs, bytes_per_elem)` — `name` is the `kernel` column in
  `bench.csv`; `bytes_per_elem` is DRAM traffic per element and is the roofline
  denominator (vecadd 3×4 B, transpose 2×4 B).
- `variant(name, kernel, block, launcher)` — the kernel symbol is passed twice
  on purpose: once as a pointer so occupancy can be reported without a launch,
  once inside the launcher because only `<<<>>>` knows the argument order.
  `name` is the `variant` column in `bench.csv`; keep existing names stable so
  old rows stay comparable.
- Run modes: `test` (all `test_shapes` × all variants, zeroed output each time),
  `bench` (banner + occupancy, then `bench_shapes` × variants, median of 50),
  `profile` (single launch of the first variant — or `profile <variant>` — at the
  largest bench shape; this is what nsys traces).
- **Buffer model:** `inputs` device inputs + one output, each `Shape::numel()`
  floats, seeded-RNG filled. Set `spec.out_numel` when the output is a different
  size (a reduction). Kernels whose *inputs* differ in size (sgemm: M×K, K×N,
  M×N) are the first thing that will need `Buffers`/`Spec` widened — do it there
  rather than reintroducing per-kernel boilerplate.

## Conventions

- 2-space indent, ~80 columns, Google-ish naming; `__restrict__` on kernel
  pointer params.
- **Comments earn their place.** One line per kernel stating the lesson (why
  this variant exists / what it costs) beats a paragraph. The mechanics are in
  `runner.cuh`; don't re-explain CUDA basics in a kernel file.
- Every kernel guards its indices — the awkward battery deliberately includes
  `0`, `1`, warp±1, block±1, primes, and ragged 2-D shapes. Use `lab::blocks()`
  for grid dims so an empty shape still yields a legal (≥1 block) launch.
- Variants are a *ladder*: each one fixes exactly one thing the previous one got
  wrong, and all of them stay in the file so the benchmark shows the delta.
- `bench.csv` is append-only across sessions and carries GPU / sm / git SHA /
  date per row. Don't rewrite it; don't rename kernels or variants casually.
- `results/peak_bw.txt` is keyed by GPU name and re-measured automatically when
  the pod hands you a different card.

## Gotchas

- `.cuda-include/` is a Mac-only copy of the CUDA headers for VS Code
  IntelliSense. It is gitignored and excluded from the rsync to the pod.
- The synced tree on the pod has no `.git`; `local/session.sh` computes
  `GIT_SHA` on the Mac and forwards it so `bench.csv` provenance stays honest.
- `local/session.sh` **merges** `results/bench.csv` on the way back rather than
  letting rsync overwrite it: a pod only ever holds the rows it produced, so a
  plain pull wipes the cross-pod history the first time you rent a new box.
  Don't simplify that back into a straight rsync.
- nsys timings are inflated by tracing — read the roofline off the `bench` run,
  never off the timeline.
- `FULL_SANITIZE=1` adds `synccheck` + `racecheck`; worth it for any kernel with
  shared memory, slow otherwise.
