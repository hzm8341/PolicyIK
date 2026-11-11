#!/usr/bin/env bash
# Simulator validity check of the Franka task sets: hold every start and goal configuration for 60 steps with
# gravity feedforward. A configuration that the simulator pushes away (contact resolution) is not a valid
# start/goal even though the offline sphere screening accepted it. Outputs in outputs/p2/simcheck/.
set -o pipefail
cd "$(dirname "$0")/../.."
source env_local.sh
export PYTHONPATH="$PWD:$PWD/tools_local:$PWD/third_party/openarm_isaac_lab${PYTHONPATH:+:$PYTHONPATH}"
D=outputs/p2; S=$D/simcheck; RUN=tools_local/run_eval.sh; F="--robot franka --controller hold --steps 60"
mkdir -p $S
$RUN $S/eval_start.log $S/eval_start.npz $F --task_file $D/data/eval_tasks.npz
$RUN $S/eval_goal.log $S/eval_goal.npz $F --task_file $D/data/simcheck_eval_goal.npz
$RUN $S/tune_start.log $S/tune_start.npz $F --task_file $D/data/tune_tasks.npz
$RUN $S/tune_goal.log $S/tune_goal.npz $F --task_file $D/data/simcheck_tune_goal.npz
echo SIMCHECK_DONE_PART
# Diagnostics for the slow outer-loop oscillation of the classical baselines on the Franka (tuning set only):
# more joint damping, and a lower task gain with a doubled budget.
G=$D/diag; mkdir -p $G; Q="--robot franka --controller qp --lead 0.15 --task_file $D/data/tune_tasks.npz"
$RUN $G/qp_k1.5_kd3.log $G/qp_k1.5_kd3.npz $Q --k 1.5 --kd_scale 3
$RUN $G/qp_k1.5_kd3_320.log $G/qp_k1.5_kd3_320.npz $Q --k 1.5 --kd_scale 3 --steps 320
$RUN $G/qp_k1_320.log $G/qp_k1_320.npz $Q --k 1 --steps 320
$RUN $G/qp_k0.5_320.log $G/qp_k0.5_320.npz $Q --k 0.5 --steps 320
echo DIAG_DONE
