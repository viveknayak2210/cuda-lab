# cuda-lab: a 70/30 workflow

The design rule: **the pod is a batch job, not a workstation.** Every
paid session is one command from your Mac that syncs, runs a fixed battery, and
pulls artifacts back.

---

## Part 1 — The paid session

### The 60-second runbook

Paste connection details into `local/pod.env` from your Mac:

```bash
./local/session.sh --boot        # first session on a fresh pod
./local/session.sh               # every session after (incl. nsys timeline)
./local/session.sh 01_vecadd     # one kernel while iterating
NSYS_TRACE=0 ./local/session.sh 01_vecadd # skip the nsys pass for a tighter loop
PROFILE=1 ./local/session.sh 01_vecadd    # also try ncu (usually blocked on RunPod)
FULL_SANITIZE=1 ./local/session.sh        # + synccheck/racecheck (shared-mem kernels)
./local/session.sh --stop        # last run of the day
```

### What runs, in order

Phases 1–2 are `bootstrap.sh` (only with `--boot`; idempotent, ~10s). The rest
are `run.sh`, every session.

| Phase | Command | Why it's in this order |
|---|---|---|
| 1. Identify | `nvidia-smi --query-gpu=compute_cap` | You cannot trust which card you got. Everything downstream keys off `sm_XX`. |
| 2. Smoke | compile + run a 3-line kernel | If this fails, stop immediately — you have a broken pod, not a broken kernel. Costs 5 seconds. |
| 3. Build | `make -j$(nproc)` | `-Xptxas -v` prints registers/thread and smem/block. **Read these.** They're your occupancy inputs and you can't get them from the profiler as directly. |
| 4. Correctness | `./bin/K test` | Awkward-size sweep: `0, 1, 31, 32, 33, 63, 64, 65, …, 1048573 (prime), 4M`. Never benchmark a kernel that hasn't passed. |
| 5. Sanitize | `compute-sanitizer` ×2 | `memcheck` → OOB/leaks. `initcheck` → reads of uninitialized device memory. `FULL_SANITIZE=1` adds `synccheck` (illegal `__syncthreads()`) and `racecheck` (shared-memory races) — racecheck is slow and only pays off on shared-memory kernels, but it finds the bugs you can't reproduce. |
| 6. Benchmark | `./bin/K bench` | Median of 50, after 5 warmups. CUDA events, not wall clock. |
| 7. Timeline | `nsys profile --trace=cuda` (default; `NSYS_TRACE=0` skips) | Whole-program view: kernel + memcpy summary, H2D/D2H vs. compute. Traces via CUPTI, needs **no** perf counters, so it works on locked-down hosts like RunPod. It's a separate traced run — never read timings off it; the clean numbers come from phase 6. |
| 8. Profile (opt-in) | `PROFILE=1` → `ncu --set basic` | Off by default — the expensive phase, installed lazily. Caveat: RunPod hosts usually block GPU counters (`ERR_NVGPUCTRPERM`); when that happens, don't fight it — the nsys timeline + roofline numbers are your analysis. |
| 9. Package | `tar czf` | One artifact to pull back. |

### Offline nvcc (the important one)

`nvcc` is ahead-of-time — it needs no GPU and no driver to produce PTX and
cubin. NVIDIA ships the toolkit for ARM64, which runs natively under Docker on
your M4:

```bash
docker build --platform=linux/arm64 -f local/Dockerfile.offline -t cudalab-offline .
./local/cudash make ARCH=86              # full type-check of the whole tree  
./local/cudash make regs K=01_vecadd     # registers/thread, smem/block
./local/cudash make sass K=01_vecadd ARCH=90a   # read Hopper SASS, no H100 needed
```

### Viewers

Nsight Systems opens the `.nsys-rep`
timeline every session produces by default; Nsight Compute (native macOS arm64
build) opens the `.ncu-rep` files from `PROFILE=1` runs on pods that allow
perf counters.

### Local setup checklist

```
[ ] Docker Desktop (for the offline nvcc container)
[ ] Nsight Compute.app + Nsight Systems.app
[ ] ~/.ssh/id_ed25519 generated, pubkey pasted into RunPod account settings
[ ] rsync (preinstalled) — verify: rsync --version
[ ] repo cloned, local/pod.env created
[ ] a scratch C++ setup for CPU reference implementations (clang++ is already there)
```

### What you write locally, before any pod exists

For each kernel, in this order:

1. **CPU reference** — the thing you'll diff against. Plain C++, no CUDA.
2. **Test input generation + expected outputs** — seeded RNG so runs are
   reproducible.
3. **Grid/block mapping on paper** — how does thread `(bx, tx)` map to data
   index? Draw it. Most first-kernel bugs are here.
4. **The `.cu` kernel** — type-check it with `./local/cudash make`.
5. **The awkward-size list** for this specific kernel — the defaults in
   `harness.cuh` cover 1-D; add your own for 2-D tiling (non-square, non-multiple
   of tile size, single row, single column).
6. **A written prediction** — arithmetic intensity, and your guess at
   %-of-peak-bandwidth. Write it in a comment. Then go measure and see how
   wrong you were. This is the highest-value habit in the whole workflow.

---