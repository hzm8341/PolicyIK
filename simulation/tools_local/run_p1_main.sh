#!/usr/bin/env bash
# P1 main runs on the 4,120 eval tasks with frozen settings (selection from the tuning set).
cd "$(dirname "$0")/.."
source env_local.sh
export PYTHONPATH="$PWD:$PWD/third_party/openarm_isaac_lab${PYTHONPATH:+:$PYTHONPATH}"
T=data/eval_tasks.npz
R=outputs/p1/eval
RUN=tools_local/run_eval.sh
read DK DL QK QL MK ML MT < <(python -c "
import json; s=json.load(open('outputs/p1/tune/selection.json')); print(s['dls']['k'], s['dls']['lead'], s['qp']['k'], s['qp']['lead'], s['mpc']['k'], s['mpc']['lead'], s['mpc']['trust'])")
echo "selected dls k=$DK lead=$DL ; qp k=$QK lead=$QL ; mpc k=$MK lead=$ML trust=$MT"
# 1) PolicyIK with joint dumps (no FF = paper setting) and with gravity FF (sensitivity)
for s in 1 2 3; do
  $RUN $R/policyik_s$s/single.log $R/policyik_s$s/single.npz --controller policyik --checkpoint checkpoints/policyik_s$s.pth --task_file $T
done
# 2) classical, frozen selection + paper settings, all with gravity FF (paper setting)
$RUN $R/dls_sel/single.log $R/dls_sel/single.npz --controller dls --k $DK --lead $DL --task_file $T
$RUN $R/qp_sel/single.log  $R/qp_sel/single.npz  --controller qp  --k $QK --lead $QL --task_file $T
$RUN $R/dls_paper/single.log $R/dls_paper/single.npz --controller dls --k 2 --lead 0.15 --task_file $T
$RUN $R/qp_paper/single.log  $R/qp_paper/single.npz  --controller qp  --k 1.5 --lead 0.15 --task_file $T
$RUN $R/mpc_sel/single.log $R/mpc_sel/single.npz --controller mpc --k $MK --lead $ML --trust $MT --task_file $T
$RUN $R/mpc_paper/single.log $R/mpc_paper/single.npz --controller mpc --k 1.5 --lead 0.15 --trust 0.1 --task_file $T
$RUN $R/dls_default/single.log $R/dls_default/single.npz --controller dls_default --task_file $T
# 3) feedforward sensitivity
for s in 1 2 3; do
  $RUN $R/policyik_ff_s$s/single.log $R/policyik_ff_s$s/single.npz --controller policyik --gravity_ff --checkpoint checkpoints/policyik_s$s.pth --task_file $T
done
$RUN $R/dls_sel_noff/single.log $R/dls_sel_noff/single.npz --controller dls --k $DK --lead $DL --no_gravity_ff --task_file $T
$RUN $R/qp_sel_noff/single.log  $R/qp_sel_noff/single.npz  --controller qp  --k $QK --lead $QL --no_gravity_ff --task_file $T
$RUN $R/dls_default_noff/single.log $R/dls_default_noff/single.npz --controller dls_default --no_gravity_ff --task_file $T
# 4) deployment protocol for the selected classical controllers (same protocol as PolicyIK)
for c in dls_sel qp_sel mpc_sel; do
  case $c in
    dls_sel) A="--controller dls --k $DK --lead $DL";;
    qp_sel)  A="--controller qp --k $QK --lead $QL";;
    mpc_sel) A="--controller mpc --k $MK --lead $ML --trust $MT";;
  esac
  python -m policyik_sim.corrected_goals --single $R/$c/single.npz --task_file $T --output $R/$c/corrected_tasks.npz
  $RUN $R/$c/corrected.log $R/$c/corrected.npz $A --task_file $R/$c/corrected_tasks.npz
  for k in 0 1 2 3 4 5 6 7; do
    $RUN $R/$c/transient$k.log $R/$c/transient$k.npz $A --task_file $T --transient_seed $k
  done
done
echo P1_MAIN_DONE
