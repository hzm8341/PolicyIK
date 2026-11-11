#!/usr/bin/env bash
# Diagnostic (not a paper protocol): double attempt budget (320 steps = 10.7 s) for the selected controllers.
cd "$(dirname "$0")/.."
source env_local.sh
T=data/eval_tasks.npz; R=outputs/p1/budget; RUN=tools_local/run_eval.sh; mkdir -p $R
$RUN $R/policyik_s1.log $R/policyik_s1.npz --controller policyik --checkpoint checkpoints/policyik_s1.pth --steps 320 --task_file $T
$RUN $R/dls_sel.log $R/dls_sel.npz --controller dls --k 2 --lead 0.15 --steps 320 --task_file $T
$RUN $R/qp_sel.log $R/qp_sel.npz --controller qp --k 1.5 --lead 0.15 --steps 320 --task_file $T
echo BUDGET_DONE
