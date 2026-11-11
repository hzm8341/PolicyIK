#!/usr/bin/env bash
# P2 step 1 on the GPU (no training): Franka FK gate, gravity-FF hold check, DLS/QP tuning on the disjoint tuning
# set, then the frozen 4,120-task Franka test set. Idempotent (existing outputs are skipped) so it can be resumed
# after tools_local/gpu_guard.sh yields the GPU.
set -o pipefail
cd "$(dirname "$0")/../.."
source env_local.sh
export PYTHONPATH="$PWD:$PWD/tools_local:$PWD/third_party/openarm_isaac_lab${PYTHONPATH:+:$PYTHONPATH}"
D=outputs/p2; RUN=tools_local/run_eval.sh
TUNE=$D/data/tune_tasks.npz; TEST=$D/data/eval_tasks.npz
F="--robot franka"

# 1. FK / limit gate (USD vs URDF) and holding-offset statistics
if [ ! -s $D/fk_check.txt ]; then
  env -u DISPLAY python tools_local/policyik_franka/check_fk_sim.py --headless > $D/fk_check.log 2>&1 || { echo "FK check crashed"; exit 1; }
fi
grep -q "FK GATE: PASS" $D/fk_check.txt || { echo "FK GATE FAILED -- stopping"; exit 2; }

# 2. hold with and without gravity feedforward (drift check of the FF path on the Franka)
$RUN $D/hold/hold_ff.log $D/hold/hold_ff.npz $F --controller hold --task_file $TUNE --steps 60
$RUN $D/hold/hold_noff.log $D/hold/hold_noff.npz $F --controller hold --no_gravity_ff --task_file $TUNE --steps 60

# 3. tuning grid on the tuning set (same protocol as P1)
for c in dls qp; do for k in 1 1.5 2 3 5; do for l in 0.05 0.1 0.15 0.25 0.5; do
  $RUN $D/tune/${c}_k${k}_lead${l}.log $D/tune/${c}_k${k}_lead${l}.npz $F --controller $c --k $k --lead $l --task_file $TUNE
done; done; done
python tools_local/summarize_tuning.py $D/tune > $D/tune/tuning_summary.txt 2>/dev/null
read DK DL QK QL < <(python -c "
import json; s=json.load(open('$D/tune/selection.json')); print(s['dls']['k'], s['dls']['lead'], s['qp']['k'], s['qp']['lead'])")
echo "selected DLS k=$DK lead=$DL, QP k=$QK lead=$QL"

# 4. frozen test set: selected, OpenArm-paper settings, Isaac Lab default DLS, MPC (QP-selected k/lead, trust 10)
E=$D/eval
$RUN $E/dls_sel.log $E/dls_sel.npz $F --controller dls --k $DK --lead $DL --task_file $TEST
$RUN $E/qp_sel.log $E/qp_sel.npz $F --controller qp --k $QK --lead $QL --task_file $TEST
$RUN $E/dls_paper.log $E/dls_paper.npz $F --controller dls --k 2 --lead 0.15 --task_file $TEST
$RUN $E/qp_paper.log $E/qp_paper.npz $F --controller qp --k 1.5 --lead 0.15 --task_file $TEST
$RUN $E/dls_default.log $E/dls_default.npz $F --controller dls_default --task_file $TEST
$RUN $E/mpc_sel.log $E/mpc_sel.npz $F --controller mpc --k $QK --lead $QL --trust 10 --task_file $TEST
# 5. double-budget diagnostic
$RUN $E/dls_sel_320.log $E/dls_sel_320.npz $F --controller dls --k $DK --lead $DL --steps 320 --task_file $TEST
$RUN $E/qp_sel_320.log $E/qp_sel_320.npz $F --controller qp --k $QK --lead $QL --steps 320 --task_file $TEST
echo P2_BASELINES_DONE
