# cuda-lab: a 70/30 workflow

The design rule: **the pod is a batch job, not a workstation.** You should never
sit at an SSH prompt on a running GPU thinking about what to type next. Every
paid session is one command from your Mac that syncs, runs a fixed battery, and
pulls artifacts back.

---

## Part 1 — The paid session

### The 60-second runbook

When RunPod hands you a pod, paste its connection details into `local/pod.env`,
then from your Mac:

```bash
./local/session.sh --boot        # first session on a fresh pod
./local/session.sh               # every session after (incl. nsys timeline)
./local/session.sh 01_vecadd     # one kernel while iterating
NSYS_TRACE=0 ./local/session.sh 01_vecadd # skip the nsys pass for a tighter loop
PROFILE=1 ./local/session.sh 01_vecadd    # also try ncu (usually blocked on RunPod)
FULL_SANITIZE=1 ./local/session.sh        # + synccheck/racecheck (shared-mem kernels)
./local/session.sh --stop        # last run of the day
```

That's it. Everything below is what those commands do, so you can debug it when
it breaks.

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

### Rules that keep the meter short

**There is no auto-stop — shutdown is on you.** End the day with
`./local/session.sh --stop`, then verify in the RunPod dashboard anyway. A
forgotten A4000 for a week is $28 — more than your entire planned month.

**Never profile and think at the same time.** Collect `.nsys-rep`/`.ncu-rep`,
kill the pod, then spend two hours reading them on the Mac in the Nsight apps.
Analysis is free; the pod is not.

**Batch your hypotheses.** Don't spin up to test one idea. Queue five kernel
variants locally, run them in one session, compare the CSV afterwards. The
per-session overhead (boot, image pull, bootstrap) is fixed, so five experiments
per session costs a fifth as much overhead as one.

**Leave profiling off while iterating on correctness.** It's opt-in
(`PROFILE=1`) precisely because it's the expensive phase; you don't need it
until the kernel is right.

> A reality check on the money: at $0.17/hr an A4000 costs **0.28 cents per
> minute**. A sloppy 40-minute session costs 11 cents. You are not really
> optimizing dollars here — you're optimizing *attention*, and removing the
> low-grade anxiety of a running meter is what actually makes you experiment
> freely. The one genuine financial risk is forgetting to shut down.

---

## Part 2 — The local half

You need three things on the Mac: an editor, an offline compiler, and the
report viewers. No simulation.

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

This kills the single biggest source of wasted paid time: discovering a missing
semicolon *after* you've started billing. Every compile error, every template
mess, every `__device__` qualifier mistake gets caught here for free. You can
also compile for `sm_90a` and read the generated SASS for TMA and `wgmma` code
months before you ever rent a Hopper card.

You cannot run anything. `cudaMalloc` fails. That is the correct trade.

### Viewers

Install both host GUIs on the Mac. Nsight Systems opens the `.nsys-rep`
timeline every session produces by default; Nsight Compute (native macOS arm64
build) opens the `.ncu-rep` files from `PROFILE=1` runs on pods that allow
perf counters. This is where you'll spend most of your actual learning hours,
at zero cost.

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

## Part 3 — Getting code onto the pod fast

Ranked by how much friction they remove:

### 1. Put the repo on a Network Volume (do this first)

A RunPod Network Volume persists across pod termination and costs $0.07/GB/month.
Put `cuda-lab` at `/workspace/cuda-lab` on a 10 GB volume — **70 cents a month** —
and you clone it exactly once, ever. Every subsequent pod mounts the volume with
your repo, your build cache, and your accumulated `results/bench.csv` already
there. This turns "set up the pod" from minutes into zero.

### 2. `rsync` over a multiplexed SSH connection

`session.sh` uses `rsync -rlptz --delete` (not `-a`: the RunPod volume forbids
`chown`, and preserving root ownership breaks the pull back to the Mac) with
`ControlMaster`/`ControlPersist`, so the TCP+auth handshake happens once and
subsequent calls reuse it. For a source tree of a few hundred KB, a sync is
well under a second. Excluding `.git`, `bin`, `build`, and `results` keeps it
that way.

```bash
# ~/.ssh/config — add this so you never paste a long ssh command again
Host runpod
    HostName 194.26.196.42      # update per pod
    User root
    Port 22041                  # update per pod
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

### 3. Git as the backbone, rsync for the last mile

Keep the repo public on GitHub (learning code doesn't need to be private) so
the pod can `git clone https://…` with **no credentials at all** — no deploy
keys, no PATs, no SSH agent forwarding. Then `git pull` is your baseline sync
and `rsync` handles uncommitted work-in-progress. Commit at the end of each
session; the network volume means you rarely even need the clone.

### 4. Continuous sync, if you want an even tighter loop

[Mutagen](https://mutagen.io) (`brew install mutagen-io/mutagen/mutagen`) does
live bidirectional sync on file save:

```bash
mutagen sync create --name=lab \
  --ignore=.git,bin,build \
  ./ root@194.26.196.42:22041:/workspace/cuda-lab
```

Now editing in your Mac editor updates the pod within milliseconds and you just
re-run `make` remotely. This is the nicest experience, at the cost of one more
moving part.

### What NOT to do

- **Don't use VS Code Remote-SSH as your primary editor.** It works, but it
  quietly encourages you to *live* on the pod — reading, thinking, and editing
  while the meter runs. That's precisely the habit this workflow exists to
  break. Use it for quick remote poking, not for authoring.
- **Don't `scp` the whole tree each time.** No delta detection; you'll re-send
  everything.
- **Don't bake code into a Docker image.** Rebuild latency per edit is far worse
  than any sync.
