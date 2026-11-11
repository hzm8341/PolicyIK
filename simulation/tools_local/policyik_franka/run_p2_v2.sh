#!/usr/bin/env bash
# P2 baselines, version 2: narrowed, simulator-validated Franka task sets (make_franka_tasks_v2.py) and three PD
# gain sets, because the paper has no Franka experiment and Kp 80 / Kd 4 is underdamped on this arm:
#   g80_4    Kp 80  / Kd 4    paper (OpenArm) values = Isaac Lab FRANKA_PANDA_CFG
#   g80_12   Kp 80  / Kd 12   same stiffness, damping x3 (v1 diagnostic)
#   g400_80  Kp 400 / Kd 80   Isaac Lab FRANKA_PANDA_HIGH_PD_CFG gains (gravity stays enabled here)
# Tuning grid per gain set on the tuning set, then the frozen test set. Idempotent / resumable.
set -o pipefail
cd "$(dirname "$0")/../.."
source env_local.sh
export PYTHONPATH="$PWD:$PWD/tools_local:$PWD/third_party/openarm_isaac_lab${PYTHONPATH:+:$PYTHONPATH}"
D=outputs/p2/data_v2; O=outputs/p2/v2; RUN=tools_local/run_eval.sh; F="--robot franka"
mkdir -p $O

# 1. simulator validity of every candidate configuration, then the split
if [ ! -s $D/eval_tasks.npz ]; then
  $RUN $O/simcheck_hold.log $D/simcheck_hold.npz $F --controller hold --steps 60 --num_envs 8192 --task_file $D/simcheck_candidates.npz || exit 1
  python tools_local/policyik_franka/make_franka_tasks_v2.py split > $O/split.txt 2>&1 || { echo "split failed"; exit 1; }
fi
cat $O/split.txt
TUNE=$D/tune_tasks.npz; TEST=$D/eval_tasks.npz

gains() {   # <name> <kp_scale> <kd_scale>: tuning grid + test set for one gain set
  local G=$1 PD="--kp_scale $2 --kd_scale $3" T E c k l DK DL QK QL
  T=$O/$G/tune; E=$O/$G/eval
  for c in dls qp; do for k in 0.5 1 1.5 2 3; do for l in 0.02 0.05 0.1 0.15 0.25; do
    $RUN $T/${c}_k${k}_lead${l}.log $T/${c}_k${k}_lead${l}.npz $F $PD --controller $c --k $k --lead $l --task_file $TUNE
  done; done; done
  python tools_local/summarize_tuning.py $T > $T/tuning_summary.txt 2>/dev/null
  read DK DL QK QL < <(python -c "
import json; s=json.load(open('$T/selection.json')); print(s['dls']['k'], s['dls']['lead'], s['qp']['k'], s['qp']['lead'])")
  echo "$G selected DLS k=$DK lead=$DL, QP k=$QK lead=$QL"
  $RUN $E/dls_sel.log $E/dls_sel.npz $F $PD --controller dls --k $DK --lead $DL --task_file $TEST
  $RUN $E/qp_sel.log $E/qp_sel.npz $F $PD --controller qp --k $QK --lead $QL --task_file $TEST
  $RUN $E/mpc_sel.log $E/mpc_sel.npz $F $PD --controller mpc --k $QK --lead $QL --trust 10 --task_file $TEST
  $RUN $E/dls_default.log $E/dls_default.npz $F $PD --controller dls_default --task_file $TEST
  $RUN $E/dls_sel_320.log $E/dls_sel_320.npz $F $PD --controller dls --k $DK --lead $DL --steps 320 --task_file $TEST
  $RUN $E/qp_sel_320.log $E/qp_sel_320.npz $F $PD --controller qp --k $QK --lead $QL --steps 320 --task_file $TEST
}
# the three gain sets run in parallel (about 3 GB of GPU memory each)
gains g80_4 1 1 > $O/g80_4.log 2>&1 &
gains g80_12 1 3 > $O/g80_12.log 2>&1 &
gains g400_80 5 20 > $O/g400_80.log 2>&1 &
wait
cat $O/g80_4.log $O/g80_12.log $O/g400_80.log | grep -E "selected|FAILED"
echo P2_V2_DONE
