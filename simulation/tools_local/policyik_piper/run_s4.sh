#!/usr/bin/env bash
# Piper stage S4, robustness part: arm PD stiffness scaled by 0.9 / 0.95 / 1.05 / 1.1 (damping nominal) on the
# 4,120 test tasks, for the three trained seeds and the tuned DLS / QP. Idempotent. Two streams side by side.
set -o pipefail
cd "$(dirname "$0")/../.."
source env_local.sh
export PYTHONPATH="$PWD:$PWD/tools_local:$PWD/third_party/openarm_isaac_lab${PYTHONPATH:+:$PYTHONPATH}"
R=outputs/piper/s4/robust; X="--robot piper --task_file outputs/piper/data/eval_tasks.npz"; mkdir -p $R
read DK DL QK QL < <(python -c "
import json; s=json.load(open('outputs/piper/s2/tune/selection.json')); print(s['dls']['k'], s['dls']['lead'], s['qp']['k'], s['qp']['lead'])")
run() {   # <name> [args...]
  local OUT=$R/$1.npz try rc n=$1; shift
  [ -s "$OUT" ] && { echo "skip (exists) $OUT"; return 0; }
  for try in 1 2 3; do
    env -u DISPLAY python tools_local/evaluate_ctrl_nj.py --headless --output "$OUT" $X "$@" > $R/$n.log 2>&1; rc=$?
    [ $rc -eq 0 ] && [ -s "$OUT" ] && { echo "ok $OUT (try $try)"; return 0; }
    echo "FAILED rc=$rc try $try: $OUT"; sleep 5
  done
  return 1
}
pol() { for KP in 0.9 0.95 1.05 1.1; do for s in "$@"; do
  run policyik_s${s}_kp$KP --controller policyik --checkpoint logs/rl_games/piper_s$s/checkpoint_average.pth --kp_scale $KP
done; done; }
cls() { for KP in 0.9 0.95 1.05 1.1; do
  run qp_sel_kp$KP --controller qp --k $QK --lead $QL --kp_scale $KP
  run dls_sel_kp$KP --controller dls --k $DK --lead $DL --kp_scale $KP
done; pol 3; }
pol 1 2 & cls & wait
echo S4_ROBUST_DONE
