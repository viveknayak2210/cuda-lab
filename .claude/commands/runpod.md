---
description: Boot and run the cuda-lab pipeline on a freshly rented RunPod instance
argument-hint: <host:port> [kernel]
---

Drive the cuda-lab workflow against a RunPod GPU. The repo lives at `~/coding/cuda`; run everything from there. The pod is a batch job, not a workstation — sync, run a fixed battery, pull artifacts, analyze on the Mac.

## Input (`$ARGUMENTS`)

- **Required — the pod's host and port**, as `HOST:PORT` or `HOST PORT`. Take these from RunPod's **"SSH over exposed TCP" / Direct TCP** entry. Always use that endpoint, never `ssh.runpod.io`: rsync needs SCP/SFTP, which the proxy endpoint does not provide.
- **Optional — a kernel to focus on**: a folder name like `03_matmul` or a path like `kernels/03_matmul/main.cu`. Reduce any path to its kernel-folder name. If the user also says "profile"/"analyze", treat profiling as requested.

## Steps

1. Parse `HOST`, `PORT`, and optional `KERNEL` from `$ARGUMENTS`.
2. Point the repo at this pod: set `HOST=` and `PORT=` in `~/coding/cuda/local/pod.env` (leave `KEY=~/.ssh/id_ed25519`).
3. Verify reachability and see the card before spending compute:
   ```
   ssh -p PORT -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
     root@HOST 'nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader'
   ```
   If it fails, stop and report — the pod isn't reachable or the key isn't registered.
4. Run the battery from the Mac (bootstrap is idempotent, so `--boot` is always safe):
   - No kernel: `./local/session.sh --boot`
   - A kernel: `./local/session.sh --boot KERNEL`
   - Also attempt an ncu report: `PROFILE=1 ./local/session.sh --boot KERNEL`
   This bootstraps the pod, builds, runs correctness → sanitizers → benchmark → **nsys timeline (by default)**, and pulls results into `./results/` — including the `.nsys-rep` and its `.nsys.txt` summary. nsys is installed on the pod automatically if it isn't already present.
5. **Profilers.** An **nsys** (Nsight Systems) CUDA-activity timeline runs by default and works on RunPod: it traces via CUPTI and needs no GPU perf counters. **ncu** (Nsight Compute) is opt-in (`PROFILE=1`) and usually **fails on RunPod** with `ERR_NVGPUCTRPERM` — a host-level driver restriction you cannot change from inside the container. If ncu fails, do not try to work around it: rely on the nsys timeline + roofline (step 6) and say so plainly.
6. **Analyze, roofline-first**, and report tightly:
   - **Correctness gate:** timings mean nothing until `test` reports all sizes passed. Never present a benchmark for a kernel that failed correctness.
   - **Registers/thread** per kernel from the BUILD `ptxas info` lines.
   - **Bandwidth & % of achievable peak** per size/variant from `results/bench.csv` and the run log. These are the *untraced* numbers — read the roofline off these, never off the nsys traced run (tracing inflates the small sizes).
   - **Bound analysis:** ~100% of peak bandwidth ⇒ memory-bound (occupancy/register tuning won't help — stop). Low bandwidth well under peak ⇒ suspect coalescing/occupancy; only then is ncu worth attempting.
   - **nsys timeline:** the kernel + memcpy summary is in `results/KERNEL.nsys.txt` (and the run log); point the user to open `results/KERNEL.nsys-rep` in **Nsight Systems.app**. Watch for H2D/D2H transfer time dwarfing kernel time — the end-to-end "is it worth offloading" signal. If `PROFILE=1` produced a `.ncu-rep`, open it in Nsight Compute.app.
7. Remind the user to **terminate the pod from the RunPod UI** when done — there is no auto-stop.

Report: GPU · correctness · sanitizer status · bandwidth/roofline table · nsys timeline (kernel + transfer summary) · your bound conclusion. Keep it concise.
