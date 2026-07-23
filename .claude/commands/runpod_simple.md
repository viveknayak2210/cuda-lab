---
description: Quick output-signature check of ONE kernel on RunPod — ~5 sizes, shape + mean (or scalar values), no benchmark
argument-hint: <kernel> [variant] [host:port]
---

The lightweight loop: build **one** kernel on the pod, run it at ~5 varying sizes, and report just what came back — output `numel` + **mean** for a tensor, or the **values themselves** for a scalar/tiny output. No correctness sweep, no sanitizer, no benchmark, no nsys, no roofline, nothing pulled to `./results/`. This is the fast "did it produce something sane" smoke check, not the full battery.

This is **not** a correctness gate: outputs are **not** compared against the CPU reference. For the awkward-size correctness sweep + roofline, use `/runpod_run`.

The pod must already be reachable and bootstrapped (nvcc + a GPU). If this is a freshly rented pod, run `/runpod_setup` first. The repo lives at `~/coding/cuda`; run everything from there.

## Input (`$ARGUMENTS`)

- **Required — the kernel**: a folder name like `01_vecadd_ai` or a path like `kernels/01_vecadd_ai/main.cu`. Reduce any path to its kernel-folder name. If none is given, ask which kernel — this command runs exactly one.
- **Optional — a variant name**: e.g. `gridstride`. Defaults to the kernel's first variant. Must match a `variant(...)` name in that `main.cu`.
- **Optional — a new `HOST:PORT`**: only if the pod changed. If given, update `local/pod.env`; otherwise reuse the existing one.

## Steps

1. Parse `KERNEL` (required), optional `VARIANT`, and optional `HOST:PORT` from `$ARGUMENTS`. If a `HOST:PORT` was given, set `HOST=`/`PORT=` in `~/coding/cuda/local/pod.env` (leave `KEY=`); otherwise leave `pod.env` as-is.
2. Run the lean path from the Mac:
   - `./local/session.sh --simple KERNEL` (first variant), or
   - `./local/session.sh --simple KERNEL VARIANT` (a specific variant).
   This syncs the tree, builds just `bin/KERNEL` on the pod, and runs `bin/KERNEL simple [VARIANT]` — which sweeps ~5 increasing sizes (sampled from the kernel's own `test_shapes`, so 1-D or 2-D as appropriate) and prints one line per size. Nothing is written to `results/` and nothing is pulled back; the printed output **is** the deliverable.
   - If the run reports a connection failure or "cannot find nvcc", the pod isn't set up — fall back to `/runpod_setup`.
   - **Do not** add `PROFILE=1` / ncu (blocked on RunPod) or the nsys battery here — the point is to stay minimal.
3. **Relay the output signature** back to the user, tightly — one row per size:
   - **Tensor output:** `in=<shape>  out numel=<N>  mean=<mean>`.
   - **Scalar / tiny output** (≤8 elements, e.g. a reduction): `in=<shape>  out numel=<N>  vals=[...]`.
   Do **not** dress these up as verified correctness or as benchmark numbers — they are unverified output means. If a mean looks like `nan`/`inf` or is wildly off, flag it as a smell worth a real `/runpod_run`, but don't over-analyze.
4. Remind the user to **terminate the pod from the RunPod UI** when done — there is no auto-stop.

Report: the kernel · variant · the ~5 size→signature rows, and nothing else. Keep it short — this command exists to be quick.
