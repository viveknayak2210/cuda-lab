# cuda-lab

A personal CUDA kernel lab built around one rule: **the GPU pod is a batch job, not a workstation.**

You author and type-check kernels offline on a Mac (no GPU needed), then each paid RunPod session is a *single command* that syncs the tree, runs a fixed battery — build → correctness → sanitizers → benchmark → profile — and pulls the artifacts back so all the actual thinking happens locally at zero cost.

The full rationale (and the money-vs-attention math) is in **[GUIDE.md](GUIDE.md)**. This README is the map.

---

## Quickstart

**Local — offline `nvcc` in Docker, no GPU:**

```bash
docker build --platform=linux/arm64 -f local/Dockerfile.offline -t cudalab-offline .
./local/cudash make ARCH=86                 # type-check the whole tree
./local/cudash make regs K=02_transpose     # registers/thread + smem/block
./local/cudash make sass K=02_transpose     # read the generated SASS
```

You cannot *run* anything locally (`cudaMalloc` fails) — that's the correct trade. Catch every syntax error, register surprise, and template mess here, for free, before the meter starts.

**On a rented pod** — paste its **Direct TCP** host/port (not `ssh.runpod.io`) into `local/pod.env`:

```bash
cp local/pod.env.example local/pod.env      # then edit HOST / PORT
./local/session.sh --boot                   # first run on a fresh pod (idempotent)
./local/session.sh                           # every run after
./local/session.sh 02_transpose              # one kernel while iterating
./local/session.sh --stop                    # end of day
```

> ⚠️ **There is no auto-stop.** `--stop` shuts the pod down, but always confirm in the RunPod dashboard — a forgotten A4000 costs more than a planned month.

Results land in `./results/`: `bench.csv`, sanitizer logs, and a `.nsys-rep` timeline per kernel.

---

## The battery (one session)

Phases 1–2 are `scripts/bootstrap.sh` (`--boot` only, ~10s); the rest are `scripts/run.sh`, every session. See GUIDE.md for the full table.

| # | Phase | What it gives you |
|---|---|---|
| 1–2 | **Identify + smoke** | Which card (`sm_XX`) you got, and that it runs at all. Fail fast on a broken pod. |
| 3 | **Build** | `-Xptxas -v` → registers/thread + smem/block, your occupancy inputs. **Read them.** |
| 4 | **Correctness** | Awkward-size sweep (0, 1, warp±1, block±1, primes, …). Nothing is benchmarked until it passes. |
| 5 | **Sanitizers** | `memcheck` + `initcheck` always; `synccheck` + `racecheck` under `FULL_SANITIZE=1`. |
| 6 | **Benchmark** | Median of 50 (CUDA events), reported as **% of measured achievable bandwidth**. |
| 7 | **nsys timeline** | *Default.* CUDA-activity trace (kernel + memcpy, transfers vs compute). Needs no perf counters. `NSYS_TRACE=0` skips. |
| 8 | **ncu profile** | *Opt-in* (`PROFILE=1`). Usually blocked on RunPod (`ERR_NVGPUCTRPERM`); fall back to the timeline + roofline. |
| 9 | **Package** | One `.tar.gz` to pull back. |

**Roofline-first:** every `%-of-peak` compares against `results/peak_bw.txt` — a bandwidth the pod *measures*, not the spec sheet. ~100% of peak ⇒ memory-bound, stop tuning. Well under peak ⇒ suspect coalescing/occupancy, and only then is a profiler worth it. Each `bench.csv` row is stamped with GPU, `sm`, git SHA, and date, so a number is still trustworthy three pods and six weeks later.

---

## Kernels

Each kernel is one self-contained `kernels/NN_name/main.cu` with three run modes — `test` (correctness), `bench` (roofline), `profile` (single launch for nsys/ncu). The Makefile auto-discovers any `kernels/*/main.cu`, so adding a kernel is just adding a folder.

| Kernel | Lesson | Variants |
|---|---|---|
| **01_vecadd** | The template every kernel copies. Elementwise, memory-bound; the roofline and the awkward-size battery. | `naive`, `gridstride` |
| **02_transpose** | Coalescing. Same memory-bound roofline, but *where* the bytes land is everything — bandwidth swings from ~⅓ of peak to ~peak. | `naive` (strided write), `tiled` (shared-mem, bank conflict), `tiled_pad` (conflict-free) |

Write the **CPU reference first, locally**, before renting anything (GUIDE.md, Part 2).

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

- **Mac** with Docker Desktop (offline `nvcc`), plus **Nsight Systems.app** (opens the `.nsys-rep`) and **Nsight Compute.app** (opens `.ncu-rep` from `PROFILE=1` runs).
- A **RunPod** account with your SSH pubkey registered, and a GPU pod (developed on an RTX A4000, `sm_86`).
- `rsync` (preinstalled on macOS).
