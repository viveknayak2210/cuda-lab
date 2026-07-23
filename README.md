# cuda-lab

Personal CUDA kernel lab focused on minimizing GPU usage.

Author and type-check kernels offline on a Mac, then each paid RunPod session starts with a *single command* that runs a fixed pipeline — build → correctness → sanitizers → benchmark → profile, then pulls artifacts back.

---

## Quickstart

**Local — offline `nvcc` in Docker:**

```bash
docker build --platform=linux/arm64 -f local/Dockerfile.offline -t cudalab-offline .
./local/cudash make ARCH=86                 # type-check the whole tree
./local/cudash make regs K=02_transpose_ai     # registers/thread + smem/block
```

Catch every syntax error, register surprise, and template mess here, for free, before the meter starts.

```bash
cp local/pod.env.example local/pod.env      # then edit HOST / PORT
./local/session.sh --boot                   # first run on a fresh pod
./local/session.sh                           # every run after
./local/session.sh 02_transpose_ai              # one kernel while iterating
./local/session.sh --simple 03_dsum_ai          # quick output check, no battery
./local/session.sh --stop                    # end of day
```

> ⚠️ **There is no auto-stop.** `--stop` shuts the pod down.

Results land in `./results/`: `bench.csv`, sanitizer logs, and a `.nsys-rep` timeline per kernel.

---

## The battery (one session)

Phases 1–2 are `scripts/bootstrap.sh` (`--boot` only, ~10s); the rest are `scripts/run.sh`, every session. See GUIDE.md for the full table.

| # | Phase | What it gives you |
|---|---|---|
| 1–2 | **Identify + smoke** | Check GPU |
| 3 | **Build** | `-Xptxas -v` → registers/thread + smem/block, your occupancy inputs. |
| 4 | **Correctness** | Awkward-size sweep (0, 1, warp±1, block±1, primes, …). |
| 5 | **Sanitizers** | `memcheck` + `initcheck` always; `synccheck` + `racecheck` under `FULL_SANITIZE=1`. |
| 6 | **Benchmark** | Median of 50 (CUDA events), reported as **% of measured achievable bandwidth**. |
| 7 | **nsys timeline** | *Default.* CUDA-activity trace (kernel + memcpy, transfers vs compute). `NSYS_TRACE=0` skips. |
| 8 | **ncu profile** | *Opt-in* (`PROFILE=1`). Usually blocked on RunPod |

Each `bench.csv` row is stamped with GPU, `sm`, git SHA, and date.

---

## Quick check (`--simple`)

Not every edit needs the full meter. `--simple` builds **one** kernel and runs it at ~5 sizes, printing just an *output signature* per size — `min / max / mean` plus a `head=[…]` preview for a tensor, or the raw values for a scalar/tiny output (a `nonfinite=N` count appears if any NaN/Inf slip in). No correctness sweep, no sanitizer, no benchmark, no nsys, and nothing pulled to `./results/` — the printed lines *are* the deliverable.

```bash
./local/session.sh --simple 03_dsum_ai          # first variant
./local/session.sh --simple 03_dsum_ai warp     # a named variant
```

It is a "did it produce something sane" smoke check, **not** a correctness gate — outputs are never compared against the CPU reference (that's the awkward-size sweep in a full run). It also needs neither a reference nor `bench_shapes`, so a kernel can be `simple`-only.

Non-float outputs — an int histogram, an fp64 reduction — declare `spec.out_dtype` (`I32` / `U32` / `I64` / `U64` / `F64`) so the buffer is sized correctly and printed with the right format; the launcher takes its typed pointer from `a.out_as<T>()`, which aborts if the type disagrees with `out_dtype`. Inputs stay float, so this is for float-in → other-type-out kernels. See `03_dsum_ai`. From Claude Code the same flow is `/runpod_simple [kernel]`.

---

## Kernels

Every kernel directory carries its authorship as a suffix: `_ai` written by Claude, `_mine` written by me. Nothing in the build keys off it — the pair just sits side by side and benchmarks head-to-head.

Each kernel is one `kernels/NN_name/main.cu` holding just the lesson — a CPU reference, the `__global__` variants, and the `Spec` that wires them up. `common/runner.cuh` supplies the rest: buffers, the awkward-size sweep, timing, and four run modes — `test` (correctness), `bench` (roofline), `profile` (single launch for nsys/ncu), and `simple` (quick output check, above). Makefile auto-discovers `kernels/*/main.cu`.

| Kernel | Lesson | Variants |
|---|---|---|
| **01_vecadd_ai** | Elementwise, memory-bound; the roofline and the awkward-size battery. | `naive`, `gridstride` |
| **02_transpose_ai** | Coalescing | `naive` (strided write), `tiled` (shared-mem, bank conflict), `tiled_pad` (conflict-free) |
| **03_dsum_ai** | Reduction with a non-float (`fp64`) output through the `simple` flow. | `shared`, `warp` |

---

## Layout

```
kernels/NN_name/main.cu   one kernel: CPU ref + __global__ variants + Spec
                          (suffix = authorship: _ai = Claude, _mine = me)
common/check.cuh          CUDA_CHECK / KERNEL_CHECK / ceil_div
common/harness.cuh        event timing, compare, peak-bw probe, CSV record
                          (w/ provenance), occupancy report
common/runner.cuh         Shape/Args/Spec, buffers, awkward-size batteries,
                          the test/bench/profile driver
Makefile                  arch auto-detect; auto-discovers kernels; ptx/sass/regs
scripts/bootstrap.sh      one-time pod setup (identify + smoke)
scripts/run.sh            the battery
scripts/simple.sh         the lean --simple path: build one kernel, print a signature
local/session.sh          Mac-side driver: sync → run → pull (→ stop)
local/cudash              run any toolchain cmd in the offline nvcc container
local/Dockerfile.offline  ARM64 CUDA toolkit, no GPU/driver needed
local/pod.env.example     connection template (copy to local/pod.env)
GUIDE.md                  the full workflow: why the pod is a batch job
CLAUDE.md                 repo contract for agents: rules, layout, conventions
.claude/commands/runpod_setup.md  Claude Code `/runpod_setup HOST:PORT` — first-time boot + run
.claude/commands/runpod_run.md    Claude Code `/runpod_run [kernel]` — repeat run, no re-bootstrap
.claude/commands/runpod_simple.md Claude Code `/runpod_simple [kernel]` — quick output check
```

---

## Requirements

- **Mac** with Docker Desktop (offline `nvcc`), plus **Nsight Systems.app**.
- A **RunPod** account with SSH pubkey registered, and a GPU pod (using RTX A4000 here).
