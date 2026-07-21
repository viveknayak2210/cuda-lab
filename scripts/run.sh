#!/usr/bin/env bash
# The whole paid session in one command. NON-INTERACTIVE by design:
# you should never be sitting at an SSH prompt thinking while this runs.
#
#   ./scripts/run.sh              # every kernel
#   ./scripts/run.sh 01_vecadd    # just one
#   SKIP_PROFILE=1 ./scripts/run.sh   # fast loop, correctness only
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
  for TOOL in memcheck initcheck synccheck racecheck; do
    # racecheck is the slow one; it is also the one that finds the bugs
    # you cannot reproduce. Do not skip it on shared-memory kernels.
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

  if [ "${SKIP_PROFILE:-0}" != "1" ]; then
    phase "PROFILE  $K"
    nsys profile --force-overwrite true -o "results/$K" \
         --stats=false "$BIN" profile >/dev/null 2>&1 \
      && echo "  nsys -> results/$K.nsys-rep" || echo "  nsys failed"

    # --set full is many replay passes but you are profiling one small launch.
    # Swap to --set basic if a kernel gets big enough that this drags.
    ncu --set full --force-overwrite -o "results/$K" \
        --target-processes all "$BIN" profile >/dev/null 2>&1 \
      && echo "  ncu  -> results/$K.ncu-rep" || echo "  ncu failed (perms?)"
  fi
done

phase "PACKAGE"
tar czf "results/session-$TS.tar.gz" \
    -C results $(cd results && ls | grep -v '\.tar\.gz$') 2>/dev/null
echo "  results/session-$TS.tar.gz  ($(du -h results/session-$TS.tar.gz | cut -f1))"

echo
echo "═══ total paid compute: $((SECONDS))s  (~\$$(awk "BEGIN{printf \"%.4f\", $SECONDS/3600*0.17}")) ═══"
[ "$FAIL" = "0" ] && echo "ALL GREEN" || echo "FAILURES PRESENT -- see log"
exit $FAIL
