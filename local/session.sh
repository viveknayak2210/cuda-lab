#!/usr/bin/env bash
# Run from your Mac. Drives an entire paid session without you ever sitting
# at an interactive SSH prompt.
#
#   ./local/session.sh              # sync, run everything, pull results
#   ./local/session.sh 01_vecadd    # one kernel
#   ./local/session.sh --boot       # include first-time bootstrap
#   ./local/session.sh --stop       # ...and stop the pod when done
#
# Requires local/pod.env with the connection details RunPod gave you:
#   HOST=194.26.196.42
#   PORT=22041
#   KEY=~/.ssh/id_ed25519
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f local/pod.env ] || { echo "create local/pod.env first (see header)"; exit 1; }
# shellcheck disable=SC1091
source local/pod.env
REMOTE=/workspace/cuda-lab
SSH=(ssh -p "$PORT" -i "${KEY/#\~/$HOME}" \
     -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
     -o ControlMaster=auto -o ControlPath=/tmp/cm-runpod -o ControlPersist=10m)

BOOT=0; STOP=0; FILTER=""
for a in "$@"; do
  case "$a" in
    --boot) BOOT=1 ;;
    --stop) STOP=1 ;;
    *) FILTER="$a" ;;
  esac
done

echo "▸ sync up"
# -rlptz (not -a): skip owner/group preservation. The RunPod volume forbids
# chown, and preserving root ownership also fails on the pull back to the Mac.
# Exclude .cuda-include (Mac-only IntelliSense headers) and editor cruft.
rsync -rlptz --delete \
  --exclude '.git' --exclude 'bin' --exclude 'build' --exclude 'results' \
  --exclude '.cuda-include' --exclude '.vscode' --exclude '.DS_Store' \
  -e "${SSH[*]}" ./ "root@$HOST:$REMOTE/"

if [ "$BOOT" = "1" ]; then
  echo "▸ bootstrap"
  "${SSH[@]}" "root@$HOST" "cd $REMOTE && chmod +x scripts/*.sh && ./scripts/bootstrap.sh"
  echo "▸ arming deadman (45m)"
  "${SSH[@]}" "root@$HOST" "cd $REMOTE && nohup ./scripts/deadman.sh 300 >/dev/null 2>&1 &" || true
fi

echo "▸ run"
"${SSH[@]}" "root@$HOST" "cd $REMOTE && ./scripts/run.sh $FILTER" || echo "(run reported failures)"

echo "▸ pull results"
mkdir -p results
rsync -rlptz -e "${SSH[*]}" "root@$HOST:$REMOTE/results/" ./results/

if [ "$STOP" = "1" ]; then
  echo "▸ stopping pod"
  "${SSH[@]}" "root@$HOST" 'runpodctl stop pod $RUNPOD_POD_ID' || \
    echo "  !! could not auto-stop -- CHECK THE DASHBOARD"
fi

echo "▸ done. reports in ./results/ -- open the .ncu-rep in Nsight Compute.app"
