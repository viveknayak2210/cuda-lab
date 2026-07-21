#!/usr/bin/env bash
# Auto-stop this pod after N minutes no matter what. Start it the moment you
# connect; a forgotten pod costs more than everything else combined.
#
#   nohup ./scripts/deadman.sh 45 &>/dev/null &
#
# Cancel with:  pkill -f deadman.sh
set -uo pipefail
MINUTES="${1:-60}"

echo "deadman armed: pod stops in ${MINUTES}m at $(date -d "+${MINUTES} minutes" '+%H:%M' 2>/dev/null || echo '?')"
sleep $((MINUTES * 60))

echo "deadman firing"
if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl stop pod "$RUNPOD_POD_ID"
elif [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
    -H 'Content-Type: application/json' \
    -d "{\"query\":\"mutation{podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}){id}}\"}"
else
  # Last resort: kill the container. RunPod usually reaps the pod after this.
  echo "no runpodctl/API key -- halting container"
  kill -9 1 2>/dev/null || shutdown -h now
fi
