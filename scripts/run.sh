#!/usr/bin/env bash
# The whole paid session in one command. NON-INTERACTIVE by design:
# you should never be sitting at an SSH prompt thinking while this runs.
#
#   ./scripts/run.sh              # build, correctness, sanitize, bench + nsys timeline
#   ./scripts/run.sh 01_vecadd_ai    # just one kernel
#   PROFILE=1 ./scripts/run.sh    # ALSO try an ncu report (blocked on RunPod: ERR_NVGPUCTRPERM)
#   NSYS_TRACE=0 ./scripts/run.sh # skip the default nsys timeline
#   FULL_SANITIZE=1 ./scripts/run.sh   # add synccheck + racecheck (shared-mem kernels)
#
# Exits nonzero if anything failed. Leaves results/session-<ts>.tar.gz.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$PATH

FILTER="${1:-}"
TS=$(date +%Y%m%d-%H%M%S)
FAIL=0
mkdir -p results bin
LOG="results/session-$TS.log"
exec > >(tee -a "$LOG") 2>&1

phase() { echo; echo "───── $1 ───── (+$((SECONDS))s)"; }

# ncu is installed lazily -- only if you actually ask to profile, and only once.
ensure_ncu() {
  command -v ncu >/dev/null 2>&1 && return 0
  echo "  installing nsight-compute (one-time, ~30-60s)..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq nsight-compute >/dev/null 2>&1 || true
  command -v ncu >/dev/null 2>&1
}

# nsys (Nsight Systems) traces the CUDA timeline via CUPTI activity -- it needs
# NO GPU perf counters, so unlike ncu it works on locked-down hosts like RunPod.
# That's why it runs by DEFAULT. Resolve it: on PATH, then the copy bundled with
# nsight-compute (present on most CUDA devel images), else install the CLI once.
NSYS_BIN=""
ensure_nsys() {
  [ -n "$NSYS_BIN" ] && return 0
  if command -v nsys >/dev/null 2>&1; then NSYS_BIN=nsys; return 0; fi
  NSYS_BIN=$(ls /opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys 2>/dev/null | head -1)
  [ -n "$NSYS_BIN" ] && return 0
  echo "  installing nsight-systems (one-time, ~30-60s)..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq nsight-systems-cli >/dev/null 2>&1 \
    || apt-get install -y -qq nsight-systems >/dev/null 2>&1 || true
  command -v nsys >/dev/null 2>&1 && { NSYS_BIN=nsys; return 0; }
  NSYS_BIN=$(ls /opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys 2>/dev/null | head -1)
  [ -n "$NSYS_BIN" ] && return 0
  return 1
}

# Default sanitize is the fast, always-relevant pair. synccheck/racecheck only
# find bugs in shared-memory kernels and racecheck is slow -- opt in for those.
SANITIZERS="memcheck initcheck"
[ "${FULL_SANITIZE:-0}" = "1" ] && SANITIZERS="memcheck initcheck synccheck racecheck"

phase "BUILD"
# Gate on make's own status, not the pipeline's: with pipefail, grep selecting
# zero lines (a silent up-to-date build) would otherwise read as BUILD FAILED.
make -j"$(nproc)" 2>&1 | grep -Ev '^\s*$'
MAKE_STATUS=${PIPESTATUS[0]}
[ "$MAKE_STATUS" = "0" ] || { echo "BUILD FAILED"; exit 1; }
# ptxas info lines above are your register/smem budget -- read them.

KERNELS=$(ls kernels)
[ -n "$FILTER" ] && KERNELS="$FILTER"

for K in $KERNELS; do
  BIN="bin/$K"
  [ -x "$BIN" ] || { echo "skip $K (no binary)"; continue; }

  phase "CORRECTNESS  $K"
  "$BIN" test || { echo "!! tests failed: $K"; FAIL=1; }

  phase "SANITIZERS  $K"
  for TOOL in $SANITIZERS; do
    printf '  %-11s ' "$TOOL"
    if compute-sanitizer --tool "$TOOL" --error-exitcode 1 \
         "$BIN" test > "results/$K.$TOOL.txt" 2>&1; then
      echo "clean"
    else
      echo "ERRORS -> results/$K.$TOOL.txt"
      FAIL=1
    fi
  done

  phase "BENCHMARK  $K"
  "$BIN" bench || FAIL=1

  # nsys runs by DEFAULT (opt out with NSYS_TRACE=0): a CUDA-activity timeline
  # that needs no perf counters, so it works where ncu is blocked (RunPod). This
  # is a SEPARATE traced run -- the clean roofline numbers come from BENCHMARK
  # above; tracing overhead here would inflate them, so don't read timings off it.
  if [ "${NSYS_TRACE:-1}" = "1" ]; then
    phase "NSYS TIMELINE  $K"
    if ensure_nsys; then
      # --trace=cuda only -- NO --gpu-metrics-device (that path DOES need the
      # perf counters ncu is denied). "$BIN profile" is the single-launch target.
      if "$NSYS_BIN" profile --trace=cuda --force-overwrite=true \
           -o "results/$K" "$BIN" profile >/dev/null 2>&1; then
        "$NSYS_BIN" stats --force-export=true \
            --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
            "results/$K.nsys-rep" 2>/dev/null \
          | grep -vE '^(Processing|Exporting|Using|Generating|SQLite)' \
          | tee "results/$K.nsys.txt"
        rm -f "results/$K.sqlite"
        echo "  nsys -> results/$K.nsys-rep  (open in Nsight Systems.app)"
      else
        echo "  nsys trace failed -- skipped (timeline only; not a correctness gate)"
      fi
    else
      echo "  nsys unavailable -- skipped"
    fi
  fi

  if [ "${PROFILE:-0}" = "1" ]; then
    phase "PROFILE  $K"
    if ensure_ncu; then
      # --set basic: the sections you read 95% of the time, a few replay passes.
      # Switch to --set full by hand when you are chasing one specific counter.
      ncu --set basic --force-overwrite -o "results/$K" \
          --target-processes all "$BIN" profile >/dev/null 2>&1 \
        && echo "  ncu -> results/$K.ncu-rep  (open in Nsight Compute.app)" \
        || echo "  ncu failed (perms? try again)"
    else
      echo "  ncu unavailable -- skipped"
    fi
  fi
done

phase "PACKAGE"
# Exclude prior sessions' tarballs AND logs -- each archive should hold this
# session's artifacts, not grow monotonically. This session's log is re-added.
tar czf "results/session-$TS.tar.gz" \
    -C results $(cd results && ls | grep -vE '\.tar\.gz$|^session-.+\.log$') \
    "session-$TS.log" 2>/dev/null
echo "  results/session-$TS.tar.gz  ($(du -h results/session-$TS.tar.gz | cut -f1))"

echo
echo "═══ total on-pod compute: $((SECONDS))s ═══"
[ "$FAIL" = "0" ] && echo "ALL GREEN" || echo "FAILURES PRESENT -- see log"
exit $FAIL
