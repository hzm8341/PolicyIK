#!/usr/bin/env bash
# Robustness (separate experiment, not a paper reproduction): arm PD gain perturbations on the 4,120 eval tasks.
cd "$(dirname "$0")/.."
source env_local.sh
T=data/eval_tasks.npz
R=outputs/p1/robust
RUN=tools_local/run_eval.sh
read DK DL QK QL < <(python -c "
import json; s=json.load(open('outputs/p1/tune/selection.json')); print(s['dls']['k'], s['dls']['lead'], s['qp']['k'], s['qp']['lead'])")
for pd in "0.5 0.5" "0.8 0.8" "1.25 1.25" "2 2" "1 0.5" "1 2"; do
  set -- $pd; KP=$1; KD=$2; tag=kp${KP}_kd${KD}
  $RUN $R/policyik_s1_$tag.log $R/policyik_s1_$tag.npz --controller policyik --checkpoint checkpoints/policyik_s1.pth --kp_scale $KP --kd_scale $KD --task_file $T
  $RUN $R/policyik_ff_s1_$tag.log $R/policyik_ff_s1_$tag.npz --controller policyik --gravity_ff --checkpoint checkpoints/policyik_s1.pth --kp_scale $KP --kd_scale $KD --task_file $T
  $RUN $R/dls_sel_$tag.log $R/dls_sel_$tag.npz --controller dls --k $DK --lead $DL --kp_scale $KP --kd_scale $KD --task_file $T
  $RUN $R/qp_sel_$tag.log $R/qp_sel_$tag.npz --controller qp --k $QK --lead $QL --kp_scale $KP --kd_scale $KD --task_file $T
done
echo ROBUST_DONE
