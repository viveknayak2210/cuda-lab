#!/usr/bin/env bash
# Run from your Mac. Drives an entire paid session without you ever sitting
# at an interactive SSH prompt.
#
#   ./local/session.sh              # sync, run everything, pull results
#   ./local/session.sh 01_vecadd_ai    # one kernel
#   ./local/session.sh --boot       # include first-time bootstrap
#   ./local/session.sh --stop       # ...and stop the pod when done
#   PROFILE=1 ./local/session.sh 01_vecadd_ai   # also try an ncu report (nsys runs either way)
#   NSYS_TRACE=0 ./local/session.sh 01_vecadd_ai # skip the default nsys timeline for a tighter loop
#   FULL_SANITIZE=1 ./local/session.sh       # add synccheck + racecheck
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
fi

echo "▸ run"
# Forward the profile/sanitize toggles into the remote shell (env doesn't cross
# ssh on its own). Both default off -> the fast path. GIT_SHA is computed here
# because the synced tree on the pod has no .git; make bakes it into the
# binaries so every bench.csv row records the source version that produced it.
GIT_SHA=$(git describe --always --dirty 2>/dev/null || echo unknown)
"${SSH[@]}" "root@$HOST" \
  "cd $REMOTE && GIT_SHA=$GIT_SHA PROFILE=${PROFILE:-0} FULL_SANITIZE=${FULL_SANITIZE:-0} NSYS_TRACE=${NSYS_TRACE:-1} ./scripts/run.sh $FILTER" \
  || echo "(run reported failures)"

echo "▸ pull results"
# Pulls everything the run wrote, including the default nsys timeline
# (results/<kernel>.nsys-rep + .nsys.txt) and any opt-in .ncu-rep.
mkdir -p results
# bench.csv is the ONE file that must survive the pull. A pod only ever holds
# the rows it produced itself, so a straight rsync silently replaces the local
# history the moment you move to a new pod. Stash it, pull, then merge: local
# rows first, then any pod row not already present (exact-line dedup, which is
# also what makes a second pull of the same pod idempotent).
BENCH=results/bench.csv
STASH=$(mktemp -d)
trap 'rm -rf "$STASH"' EXIT
[ -f "$BENCH" ] && cp "$BENCH" "$STASH/local.csv"

rsync -rlptz -e "${SSH[*]}" "root@$HOST:$REMOTE/results/" ./results/

if [ -s "$STASH/local.csv" ] && [ -f "$BENCH" ]; then
  { cat "$STASH/local.csv"; tail -n +2 "$BENCH"; } \
    | awk 'NR==1 || !seen[$0]++' > "$STASH/merged.csv"
  mv "$STASH/merged.csv" "$BENCH"
  echo "  bench.csv: $(($(wc -l < "$BENCH") - 1)) rows total"
fi

if [ "$STOP" = "1" ]; then
  echo "▸ stopping pod"
  "${SSH[@]}" "root@$HOST" 'runpodctl stop pod $RUNPOD_POD_ID' || \
    echo "  !! could not auto-stop -- CHECK THE DASHBOARD"
fi

echo "▸ done. results in ./results/ -- open the .nsys-rep in Nsight Systems.app (.ncu-rep in Nsight Compute.app if you ran PROFILE=1)."
