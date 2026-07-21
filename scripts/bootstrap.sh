#!/usr/bin/env bash
# Run ONCE per fresh pod. Idempotent -- safe to re-run. Kept deliberately tiny:
# just prove we have a working GPU + compiler. Profilers are installed lazily by
# run.sh: nsys (the default timeline) and ncu (opt-in), so boot stays ~10s.
set -uo pipefail
T0=$SECONDS

echo "=== 1. what did we actually get? ==============================="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
           --format=csv,noheader || { echo "NO GPU VISIBLE -- wrong pod type?"; exit 1; }

export PATH=/usr/local/cuda/bin:$PATH
echo "nvcc: $(nvcc --version | tail -1)"

echo
echo "=== 2. sanity: does anything actually run? ====================="
mkdir -p results bin
cat > /tmp/_smoke.cu <<'EOF'
#include <cstdio>
__global__ void k(){ if(!threadIdx.x) printf("smoke ok from block %d\n", blockIdx.x); }
int main(){ k<<<1,32>>>(); return cudaDeviceSynchronize()!=cudaSuccess; }
EOF
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
nvcc -arch=sm_"$ARCH" /tmp/_smoke.cu -o /tmp/_smoke && /tmp/_smoke \
  || { echo "SMOKE TEST FAILED -- stop and debug before spending more"; exit 1; }

git config --global --add safe.directory "$(pwd)" 2>/dev/null || true

echo
echo "=== bootstrap done in $((SECONDS-T0))s | arch=sm_$ARCH ========="
echo "next:  ./scripts/run.sh          (fast loop + nsys timeline by default)"
echo "       PROFILE=1 ./scripts/run.sh (also try an ncu report -- blocked on RunPod)"
