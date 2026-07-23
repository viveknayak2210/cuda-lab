#!/usr/bin/env bash
# The bare-minimum on-pod loop: build ONE kernel and print an output signature
# at ~5 sizes -- shape + mean for a tensor, the raw values for a scalar. No
# correctness sweep, no sanitizer, no benchmark, no nsys, no packaging. This is
# the fast "did it produce something sane" check, NOT the battery in run.sh, and
# NOT a correctness gate (outputs are never compared to the CPU reference).
#
#   ./scripts/simple.sh 01_vecadd_ai              # first variant
#   ./scripts/simple.sh 01_vecadd_ai gridstride   # a specific variant
set -uo pipefail
export PATH=/usr/local/cuda/bin:$PATH

K="${1:-}"
VARIANT="${2:-}"
[ -n "$K" ] || { echo "usage: $0 <kernel> [variant]"; exit 2; }
[ -d "kernels/$K" ] || { echo "no such kernel: kernels/$K"; exit 2; }

mkdir -p bin results

echo "───── BUILD  $K ─────"
# Gate on make's own status: a silent up-to-date build yields no grep lines,
# which under pipefail would otherwise read as a failure.
make "bin/$K" 2>&1 | grep -Ev '^\s*$'
[ "${PIPESTATUS[0]}" = "0" ] || { echo "BUILD FAILED"; exit 1; }

echo "───── SIMPLE  $K ─────"
# shellcheck disable=SC2086 -- VARIANT is a single optional token, intentional split.
"bin/$K" simple $VARIANT
