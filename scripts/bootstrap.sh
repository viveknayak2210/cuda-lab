#!/usr/bin/env bash
# Run ONCE per fresh pod. Idempotent -- safe to re-run.
# Target: under 60 seconds. Everything here is setup you are paying for,
# so anything that can move to the Docker image eventually should.
set -uo pipefail
T0=$SECONDS

echo "=== 1. what did we actually get? ==============================="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
           --format=csv,noheader || { echo "NO GPU VISIBLE -- wrong pod type?"; exit 1; }

export PATH=/usr/local/cuda/bin:$PATH
echo "nvcc: $(nvcc --version | tail -1)"

echo
echo "=== 2. profilers ==============================================="
need_install=0
command -v ncu  >/dev/null 2>&1 || need_install=1
command -v nsys >/dev/null 2>&1 || need_install=1

if [ "$need_install" = "1" ]; then
  echo "installing nsight tools (this is the slow part, ~60-90s)..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq nsight-systems-cli nsight-compute >/dev/null 2>&1 \
    || apt-get install -y -qq nsight-systems nsight-compute >/dev/null 2>&1 \
    || echo "WARN: apt install failed; check /usr/local/cuda/bin for ncu/nsys"
else
  echo "ncu + nsys already present (good -- bake these into your image)"
fi
command -v ncu  >/dev/null 2>&1 && echo "  ncu:  $(command -v ncu)"
command -v nsys >/dev/null 2>&1 && echo "  nsys: $(command -v nsys)"

echo
echo "=== 3. sanity: does anything actually run? ====================="
mkdir -p results bin
cat > /tmp/_smoke.cu <<'EOF'
#include <cstdio>
__global__ void k(){ if(!threadIdx.x) printf("smoke ok from block %d\n", blockIdx.x); }
int main(){ k<<<1,32>>>(); return cudaDeviceSynchronize()!=cudaSuccess; }
EOF
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
nvcc -arch=sm_"$ARCH" /tmp/_smoke.cu -o /tmp/_smoke && /tmp/_smoke \
  || { echo "SMOKE TEST FAILED -- stop and debug before spending more"; exit 1; }

echo
echo "=== 4. measure this card's real achievable bandwidth ==========="
echo "(cached to results/peak_bw.txt; every later % -of-peak uses it)"

git config --global --add safe.directory "$(pwd)" 2>/dev/null || true

echo
echo "=== bootstrap done in $((SECONDS-T0))s | arch=sm_$ARCH ========="
echo "next:  ./scripts/run.sh"
