#!/usr/bin/env bash
# The whole paid session in one command. NON-INTERACTIVE by design:
# you should never be sitting at an SSH prompt thinking while this runs.
#
#   ./scripts/run.sh              # fast: build, correctness, sanitize, bench
#   ./scripts/run.sh 01_vecadd    # just one kernel
#   PROFILE=1 ./scripts/run.sh    # also capture an ncu (Nsight Compute) report
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

# Default sanitize is the fast, always-relevant pair. synccheck/racecheck only
# find bugs in shared-memory kernels and racecheck is slow -- opt in for those.
SANITIZERS="memcheck initcheck"
[ "${FULL_SANITIZE:-0}" = "1" ] && SANITIZERS="memcheck initcheck synccheck racecheck"

phase "BUILD"
make -j"$(nproc)" 2>&1 | grep -Ev '^\s*$' || { echo "BUILD FAILED"; exit 1; }
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
tar czf "results/session-$TS.tar.gz" \
    -C results $(cd results && ls | grep -v '\.tar\.gz$') 2>/dev/null
echo "  results/session-$TS.tar.gz  ($(du -h results/session-$TS.tar.gz | cut -f1))"

echo
echo "═══ total on-pod compute: $((SECONDS))s ═══"
[ "$FAIL" = "0" ] && echo "ALL GREEN" || echo "FAILURES PRESENT -- see log"
exit $FAIL
