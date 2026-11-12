#!/usr/bin/env bash
# Piper stage S3: train one seed with the paper's recipe, only the robot changed -- the three PPO segments and
# the checkpoint average of scripts/train.sh -- then evaluate the averaged policy on the tuning and test sets
# (no gravity feedforward, as in the paper). A finished segment is skipped; an interrupted one starts over.
# usage: bash tools_local/policyik_piper/run_s3.sh <seed>
set -o pipefail
cd "$(dirname "$0")/../.."
source env_local.sh
export PYTHONPATH="$PWD:$PWD/tools_local:$PWD/third_party/openarm_isaac_lab${PYTHONPATH:+:$PYTHONPATH}"
SEED="${1:?usage: run_s3.sh <seed>}"; O=outputs/piper/s3; D=outputs/piper/data; N=piper_s$SEED
mkdir -p $O

last() {      # <run name> <epoch>: the periodic checkpoint saved at that epoch (newest run directory)
  ls logs/rl_games/"$1"/*/nn/last_"$1"_ep_"$2"_rew_*.pth 2>/dev/null | grep -E "_rew_-?[0-9.]+\.pth$" | tail -n 1
}
segment() {   # <run name> <final epoch> [checkpoint to resume]
  [ -n "$(last "$1" "$2")" ] && { echo "trained: $(last "$1" "$2")"; return 0; }
  echo "$(date '+%F %T') training $1 to epoch $2 ${3:+from $3}"
  env -u DISPLAY python tools_local/policyik_piper/train_piper.py --task PolicyIK-Piper --headless --num_envs 4096 \
      --max_iterations "$2" --seed "$SEED" ${3:+--checkpoint "$3"} agent.params.config.name="$1" > $O/train_$1.log 2>&1
  [ -n "$(last "$1" "$2")" ] || { echo "training $1 failed (see $O/train_$1.log)"; return 1; }
}
segment ${N}_ep3500 3500 || exit 1
segment ${N}_ep8500 8500 "$(last ${N}_ep3500 3500)" || exit 1
segment $N 13000 "$(last ${N}_ep8500 8500)" || exit 1
AVG=logs/rl_games/$N/checkpoint_average.pth
[ -s $AVG ] || python -m policyik_sim.average_checkpoints --run_dir logs/rl_games/$N --out $AVG > $O/average_$N.log 2>&1 \
  || { echo "checkpoint average failed"; exit 1; }
echo "$(date '+%F %T') averaged: $AVG"
for s in tune eval; do
  OUT=$O/${N}_$s.npz
  [ -s $OUT ] || env -u DISPLAY python tools_local/evaluate_ctrl_nj.py --headless --output $OUT --robot piper \
      --controller policyik --checkpoint $AVG --task_file $D/${s}_tasks.npz > $O/${N}_$s.log 2>&1 || echo "eval $s FAILED"
done
echo S3_DONE
