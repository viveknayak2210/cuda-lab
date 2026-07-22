---
description: First-time boot of a freshly rented RunPod pod, then run the cuda-lab pipeline
argument-hint: <host:port> [kernel]
---

Drive the cuda-lab workflow against a **freshly rented** RunPod GPU: point the repo at the pod, bootstrap it once, run a fixed battery, pull artifacts, analyze on the Mac. The repo lives at `~/coding/cuda`; run everything from there. The pod is a batch job, not a workstation.

Use this the **first time** you talk to a given pod. For repeat runs against a pod that's already bootstrapped, use `/runpod_run` instead — it skips the one-time setup.

## Input (`$ARGUMENTS`)

- **Required — the pod's host and port**, as `HOST:PORT` or `HOST PORT`. Take these from RunPod's **"SSH over exposed TCP" / Direct TCP** entry. Always use that endpoint, never `ssh.runpod.io`: the proxy endpoint only gives you an interactive shell, not the pod's full SSH server that the `rsync`/`scp` file transfer in this workflow needs.
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
4. Run the battery from the Mac **with `--boot`** (bootstrap is idempotent, so `--boot` is always safe on first contact):
   - No kernel: `./local/session.sh --boot`
   - A kernel: `./local/session.sh --boot KERNEL`
   This bootstraps the pod, builds, runs correctness → sanitizers → benchmark → **nsys timeline (by default)**, and pulls results into `./results/` — including the `.nsys-rep` and its `.nsys.txt` summary. nsys is installed on the pod automatically if it isn't already present.
5. **Profilers.** An **nsys** (Nsight Systems) CUDA-activity timeline runs by default and works on RunPod: it traces via CUPTI and needs no GPU perf counters. **Do not run ncu** (Nsight Compute) on RunPod — it fails with `ERR_NVGPUCTRPERM`, a host-level driver restriction you cannot change from inside the container, so `PROFILE=1` just burns compute for no report. Rely on the nsys timeline + roofline (step 7). Only run ncu if the user explicitly asks and knows it will likely fail.
6. **Open the pulled timeline in the GUI** so it's loading while you analyze. The report viewer, not the "Select target for profiling" capture screen, is what you want — opening the file skips that screen entirely:
   ```
   open -a "NVIDIA Nsight Systems" results/KERNEL.nsys-rep 2>/dev/null || open results/KERNEL.nsys-rep
   ```
   The app bundle is `/Applications/NVIDIA Nsight Systems.app`; the `|| open` fallback covers a differently-named install. If the run had no `KERNEL`, open each `results/*.nsys-rep` the battery just produced (or just the ones the user cares about).
7. **Analyze, roofline-first**, and report tightly:
   - **Correctness gate:** timings mean nothing until `test` reports all sizes passed. Never present a benchmark for a kernel that failed correctness.
   - **Registers/thread** per kernel from the BUILD `ptxas info` lines.
   - **Bandwidth & % of achievable peak** per size/variant from `results/bench.csv` and the run log. These are the *untraced* numbers — read the roofline off these, never off the nsys traced run (tracing inflates the small sizes).
   - **Bound analysis:** ~100% of peak bandwidth ⇒ memory-bound (occupancy/register tuning won't help — stop). Low bandwidth well under peak ⇒ suspect coalescing/occupancy; read that off the nsys timeline and the ptxas/occupancy numbers (ncu would confirm it but is blocked on RunPod).
   - **nsys timeline:** the kernel + memcpy summary is in `results/KERNEL.nsys.txt` (and the run log); the `.nsys-rep` is already open in Nsight Systems from step 6. Watch for H2D/D2H transfer time dwarfing kernel time — the end-to-end "is it worth offloading" signal.
8. Remind the user to **terminate the pod from the RunPod UI** when done — there is no auto-stop.

Report: GPU · correctness · sanitizer status · bandwidth/roofline table · nsys timeline (kernel + transfer summary) · your bound conclusion. Keep it concise.
