---
description: Re-run the cuda-lab battery on an already-bootstrapped RunPod pod (no first-time setup)
argument-hint: [kernel] [host:port]
---

Fast repeat loop against a RunPod pod that has **already been bootstrapped** by `/runpod_setup` in a prior run. Skips the one-time work (nvidia-smi check, nsys install, smoke test) and just syncs → runs → pulls → analyzes. The repo lives at `~/coding/cuda`; run everything from there.

If this pod has never been set up (fresh rental, or `local/pod.env` doesn't point at it yet), use `/runpod_setup` instead — the battery will fail without the bootstrap.

## Input (`$ARGUMENTS`)

- **Optional — a kernel to focus on**: a folder name like `03_matmul` or a path like `kernels/03_matmul/main.cu`. Reduce any path to its kernel-folder name. Omit to run the whole battery.
- **Optional — a new `HOST:PORT`**: only if the pod changed since setup. If given, update `local/pod.env`. If omitted, **reuse the existing `pod.env`** — it already points at the pod you set up.

## Steps

1. Parse optional `KERNEL` and optional `HOST:PORT` from `$ARGUMENTS`. If a `HOST:PORT` was given, set `HOST=`/`PORT=` in `~/coding/cuda/local/pod.env` (leave `KEY=`); otherwise leave `pod.env` as-is.
2. Run the battery from the Mac **without `--boot`** — the pod is already provisioned, so bootstrap is pointless:
   - No kernel: `./local/session.sh`
   - A kernel: `./local/session.sh KERNEL`
   This syncs the tree, builds, runs correctness → sanitizers → benchmark → **nsys timeline (by default)**, and pulls results into `./results/`. If the run reports a connection failure or "cannot find nvcc/nsys", the pod isn't actually set up — fall back to `/runpod_setup`.
   - **Do not run ncu / `PROFILE=1`** on RunPod: it fails with `ERR_NVGPUCTRPERM` and just burns compute. nsys is enough.
3. **Open the pulled timeline in the GUI** (report viewer — not the "Select target for profiling" capture screen; opening the file skips it):
   ```
   open -a "NVIDIA Nsight Systems" results/KERNEL.nsys-rep 2>/dev/null || open results/KERNEL.nsys-rep
   ```
   The app bundle is `/Applications/NVIDIA Nsight Systems.app`. If no `KERNEL`, open each `results/*.nsys-rep` the battery just produced.
4. **Analyze, roofline-first**, and report tightly:
   - **Correctness gate:** timings mean nothing until `test` reports all sizes passed. Never present a benchmark for a kernel that failed correctness.
   - **Registers/thread** per kernel from the BUILD `ptxas info` lines.
   - **Bandwidth & % of achievable peak** per size/variant from `results/bench.csv` and the run log. These are the *untraced* numbers — read the roofline off these, never off the nsys traced run (tracing inflates the small sizes).
   - **Bound analysis:** ~100% of peak bandwidth ⇒ memory-bound (occupancy/register tuning won't help — stop). Low bandwidth well under peak ⇒ suspect coalescing/occupancy; read that off the nsys timeline and the ptxas/occupancy numbers (ncu would confirm it but is blocked on RunPod).
   - **nsys timeline:** the kernel + memcpy summary is in `results/KERNEL.nsys.txt` (and the run log); the `.nsys-rep` is already open in Nsight Systems from step 3. Watch for H2D/D2H transfer time dwarfing kernel time — the end-to-end "is it worth offloading" signal.
5. Remind the user to **terminate the pod from the RunPod UI** when done — there is no auto-stop.

Report: correctness · sanitizer status · bandwidth/roofline table · nsys timeline (kernel + transfer summary) · your bound conclusion. Keep it concise.
