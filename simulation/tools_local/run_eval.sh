#!/usr/bin/env bash
# Run one evaluate_ctrl.py job with retries on Isaac Sim startup crashes. usage: run_eval.sh <log> <output.npz> [args...]
set -o pipefail
cd "$(dirname "$0")/.."
source env_local.sh
export PYTHONPATH="$PWD/third_party/openarm_isaac_lab${PYTHONPATH:+:$PYTHONPATH}"
LOG="$1"; OUT="$2"; shift 2
mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT")"
[ -s "$OUT" ] && { echo "skip (exists) $OUT"; exit 0; }
for try in 1 2 3 4; do
  env -u DISPLAY python tools_local/evaluate_ctrl.py --headless --output "$OUT" "$@" > "$LOG" 2>&1
  rc=$?
  [ $rc -eq 0 ] && [ -s "$OUT" ] && { echo "ok $OUT (try $try)"; exit 0; }
  echo "FAILED rc=$rc try $try: $OUT" >&2
  sleep 5
done
exit 1
