#!/usr/bin/env bash
# Piper stage S2: classical baselines at the frozen gains (Kp 80 / Kd 4, gravity feedforward). Tuning grid of P1
# (k x lead) on the tuning set, selection by strict success, then the 4,120 test tasks. The DLS and QP streams
# run side by side (about 3 GB of GPU memory each). Idempotent / resumable.
set -o pipefail
cd "$(dirname "$0")/../.."
source env_local.sh
export PYTHONPATH="$PWD:$PWD/tools_local:$PWD/third_party/openarm_isaac_lab${PYTHONPATH:+:$PYTHONPATH}"
D=outputs/piper/data; T=outputs/piper/s2/tune; E=outputs/piper/s2/eval
mkdir -p $T $E

run() {   # <log> <output> [args...]
  local LOG=$1 OUT=$2 try rc; shift 2
  [ -s "$OUT" ] && { echo "skip (exists) $OUT"; return 0; }
  for try in 1 2 3; do
    env -u DISPLAY python tools_local/evaluate_ctrl_nj.py --headless --output "$OUT" --robot piper "$@" > "$LOG" 2>&1; rc=$?
    [ $rc -eq 0 ] && [ -s "$OUT" ] && { echo "ok $OUT (try $try)"; return 0; }
    echo "FAILED rc=$rc try $try: $OUT"; sleep 5
  done
  return 1
}
grid() {   # <controller>
  local c=$1 k l
  for k in 1 1.5 2 3 5; do for l in 0.05 0.1 0.15 0.25 0.5; do
    run $T/${c}_k${k}_lead${l}.log $T/${c}_k${k}_lead${l}.npz --controller $c --k $k --lead $l --task_file $D/tune_tasks.npz
  done; done
}
grid dls & grid qp & wait
python tools_local/summarize_tuning.py $T > $T/tuning_summary.txt || { echo "summary failed"; exit 1; }
read DK DL QK QL < <(python -c "
import json; s=json.load(open('$T/selection.json')); print(s['dls']['k'], s['dls']['lead'], s['qp']['k'], s['qp']['lead'])")
echo "selected DLS k=$DK lead=$DL, QP k=$QK lead=$QL"
X="--task_file $D/eval_tasks.npz"
stream_a() {
  run $E/dls_sel.log $E/dls_sel.npz $X --controller dls --k $DK --lead $DL
  run $E/dls_paper.log $E/dls_paper.npz $X --controller dls
  run $E/dls_sel_320.log $E/dls_sel_320.npz $X --controller dls --k $DK --lead $DL --steps 320
  run $E/dls_default.log $E/dls_default.npz $X --controller dls_default
}
stream_b() {
  run $E/qp_sel.log $E/qp_sel.npz $X --controller qp --k $QK --lead $QL
  run $E/qp_paper.log $E/qp_paper.npz $X --controller qp
  run $E/qp_sel_320.log $E/qp_sel_320.npz $X --controller qp --k $QK --lead $QL --steps 320
}
stream_a & stream_b & wait
echo S2_RUNS_DONE
