#!/usr/bin/env bash
# Fine stiffness-only sweep (damping nominal) for the PolicyIK checkpoint; seeds 1 and 2.
cd "$(dirname "$0")/.."
source env_local.sh
T=data/eval_tasks.npz; R=outputs/p1/robust; RUN=tools_local/run_eval.sh
for s in 1 2; do for KP in 0.9 0.95 1.05 1.1; do tag=kp${KP}_kd1
  $RUN $R/policyik_s${s}_$tag.log $R/policyik_s${s}_$tag.npz --controller policyik --checkpoint checkpoints/policyik_s$s.pth --kp_scale $KP --kd_scale 1 --task_file $T
done; done
echo KPSWEEP_DONE
