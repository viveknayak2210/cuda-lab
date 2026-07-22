# cuda-lab

Personal CUDA kernel lab focused on minimizing GPU usage.

Author and type-check kernels offline on a Mac, then each paid RunPod session starts with a *single command* that runs a fixed pipeline — build → correctness → sanitizers → benchmark → profile, then pulls artifacts back.

---

## Quickstart

**Local — offline `nvcc` in Docker:**

```bash
docker build --platform=linux/arm64 -f local/Dockerfile.offline -t cudalab-offline .
./local/cudash make ARCH=86                 # type-check the whole tree
./local/cudash make regs K=02_transpose     # registers/thread + smem/block
```

Catch every syntax error, register surprise, and template mess here, for free, before the meter starts.

```bash
cp local/pod.env.example local/pod.env      # then edit HOST / PORT
./local/session.sh --boot                   # first run on a fresh pod
./local/session.sh                           # every run after
./local/session.sh 02_transpose              # one kernel while iterating
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

## Kernels

Each kernel is one self-contained `kernels/NN_name/main.cu` with three run modes — `test` (correctness), `bench` (roofline), `profile` (single launch for nsys/ncu). Makefile auto-discovers `kernels/*/main.cu`.

| Kernel | Lesson | Variants |
|---|---|---|
| **01_vecadd** | Elementwise, memory-bound; the roofline and the awkward-size battery. | `naive`, `gridstride` |
| **02_transpose** | Coalescing | `naive` (strided write), `tiled` (shared-mem, bank conflict), `tiled_pad` (conflict-free) |

---

## Layout

```
kernels/NN_name/main.cu   one kernel: CPU ref + variants + test/bench/profile
common/check.cuh          CUDA_CHECK / KERNEL_CHECK / ceil_div
common/harness.cuh        awkward sizes, event timing, compare, peak-bw probe,
                          CSV record (w/ provenance), occupancy report
Makefile                  arch auto-detect; auto-discovers kernels; ptx/sass/regs
scripts/bootstrap.sh      one-time pod setup (identify + smoke)
scripts/run.sh            the battery
local/session.sh          Mac-side driver: sync → run → pull (→ stop)
local/cudash              run any toolchain cmd in the offline nvcc container
local/Dockerfile.offline  ARM64 CUDA toolkit, no GPU/driver needed
local/pod.env.example     connection template (copy to local/pod.env)
GUIDE.md                  the full workflow: why the pod is a batch job
.claude/commands/runpod_setup.md  Claude Code `/runpod_setup HOST:PORT` — first-time boot + run
.claude/commands/runpod_run.md    Claude Code `/runpod_run [kernel]` — repeat run, no re-bootstrap
```

---

## Requirements

- **Mac** with Docker Desktop (offline `nvcc`), plus **Nsight Systems.app**.
- A **RunPod** account with SSH pubkey registered, and a GPU pod (using RTX A4000 here).
