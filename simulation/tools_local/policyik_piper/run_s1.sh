#!/usr/bin/env bash
# Piper stage S0 (GPU part) and S1: OpenArm regression of the joint-count-generic evaluator, FK gate, hold
# tests, self-collision sampling, smoke run. Idempotent: a step whose output exists is skipped, so a run that
# gpu_guard.sh interrupted resumes. usage: tools_local/gpu_guard.sh <log> bash tools_local/policyik_piper/run_s1.sh
set -o pipefail
cd "$(dirname "$0")/../.."
source env_local.sh
export PYTHONPATH="$PWD:$PWD/tools_local:$PWD/third_party/openarm_isaac_lab${PYTHONPATH:+:$PYTHONPATH}"
O=outputs/piper/s1; R=outputs/piper/regress; D=outputs/piper/data
mkdir -p $O $R

run() {   # <script> <log> <output> [args...]: one evaluator job, retried on Isaac Sim startup crashes
  local PY=$1 LOG=$2 OUT=$3 try rc; shift 3
  [ -s "$OUT" ] && { echo "skip (exists) $OUT"; return 0; }
  for try in 1 2 3; do
    env -u DISPLAY python $PY --headless --output "$OUT" "$@" > "$LOG" 2>&1; rc=$?
    [ $rc -eq 0 ] && [ -s "$OUT" ] && { echo "ok $OUT (try $try)"; return 0; }
    echo "FAILED rc=$rc try $try: $OUT"; sleep 5
  done
  return 1
}
NJ=tools_local/evaluate_ctrl_nj.py

# 1. PG0: the generic evaluator must reproduce the current one bit for bit on the OpenArm (two controllers)
A="--robot openarm --task_file outputs/p1/tune_tasks.npz"
run tools_local/evaluate_ctrl.py $R/orig_dls.log $R/orig_dls.npz $A --controller dls --k 2 --lead 0.15 || exit 1
run $NJ $R/nj_dls.log $R/nj_dls.npz $A --controller dls --k 2 --lead 0.15 || exit 1
run tools_local/evaluate_ctrl.py $R/orig_dls2.log $R/orig_dls2.npz $A --controller dls --k 2 --lead 0.15 || exit 1
run tools_local/evaluate_ctrl.py $R/orig_pik.log $R/orig_pik.npz $A --controller policyik --checkpoint checkpoints/policyik_s1.pth || exit 1
run $NJ $R/nj_pik.log $R/nj_pik.npz $A --controller policyik --checkpoint checkpoints/policyik_s1.pth || exit 1
python tools_local/policyik_piper/compare_npz.py $R/orig_dls.npz $R/orig_dls2.npz $R/orig_dls.npz $R/nj_dls.npz \
  $R/orig_pik.npz $R/nj_pik.npz > $R/regress.txt 2>&1; echo "regress rc=$?"; cat $R/regress.txt

# 2. FK gate
if [ ! -s $O/fk_check.txt ]; then
  env -u DISPLAY python tools_local/policyik_piper/check_fk_sim.py --headless > $O/fk_check.log 2>&1 || echo "FK check FAILED rc=$?"
fi
cat $O/fk_check.txt

# 3. hold tests on the tuning starts: with / without gravity feedforward; then with self-collision enabled on
#    the tuning starts and the starts of the test set (self-collision sampling)
P="--robot piper --controller hold"
run $NJ $O/hold_ff.log $O/hold_ff.npz $P --task_file $D/tune_tasks.npz || exit 1
run $NJ $O/hold_noff.log $O/hold_noff.npz $P --no_gravity_ff --task_file $D/tune_tasks.npz || exit 1
export PIPER_SELF_COLLISIONS=1
run $NJ $O/hold_ff_selfcol.log $O/hold_ff_selfcol.npz $P --steps 60 --task_file $D/tune_tasks.npz || exit 1
run $NJ $O/hold_ff_selfcol_evalstart.log $O/hold_ff_selfcol_evalstart.npz $P --steps 60 --num_envs 4120 --task_file $D/eval_tasks.npz || exit 1
unset PIPER_SELF_COLLISIONS

# 4. smoke: DLS and QP at the paper's settings on the tuning set (finite values, rough level)
run $NJ $O/smoke_dls.log $O/smoke_dls.npz --robot piper --controller dls --task_file $D/tune_tasks.npz || exit 1
run $NJ $O/smoke_qp.log $O/smoke_qp.npz --robot piper --controller qp --task_file $D/tune_tasks.npz || exit 1
echo S1_RUNS_DONE
