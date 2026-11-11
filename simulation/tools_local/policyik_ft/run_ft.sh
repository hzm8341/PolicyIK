#!/usr/bin/env bash
# Step 4: fine-tune the released OpenArm checkpoint (s1) for EPOCHS PPO epochs in three variants (Base control,
# gravity FF, PD-gain randomization), then evaluate each on the 4,120 test tasks, nominal and under Kp/Kd changes.
# Resumable: training restarts from the newest periodic checkpoint; existing evaluation outputs are skipped.
set -o pipefail
cd "$(dirname "$0")/../.."
source env_local.sh
export PYTHONPATH="$PWD:$PWD/tools_local:$PWD/third_party/openarm_isaac_lab${PYTHONPATH:+:$PYTHONPATH}"
EPOCHS=${EPOCHS:-1000}
INIT=outputs/ft/policyik_s1_full.pth
[ -s $INIT ] || python tools_local/policyik_ft/make_full_ckpt.py checkpoints/policyik_s1.pth $INIT
T=data/eval_tasks.npz; E=outputs/ft/eval; RUN=tools_local/run_eval.sh

latest() {  # <name>: newest periodic checkpoint "epoch path", or nothing
  ls logs/rl_games/$1/*/nn/last_$1_ep_*_rew_*.pth 2>/dev/null |
    sed -E 's/.*_ep_([0-9]+)_rew_.*/\1 &/' | sort -n | tail -n 1
}
train() {   # <variant> <name>
  local ep path
  read ep path < <(latest $2)
  if [ -n "$ep" ] && [ "$ep" -ge "$EPOCHS" ]; then echo "trained: $path"; return 0; fi
  local from=${path:-$INIT}
  echo "training $2 from $from (epoch ${ep:-0} -> $EPOCHS)"
  env -u DISPLAY python tools_local/train_ft.py --task PolicyIK-OpenArm-FT-$1 --headless --num_envs 4096 \
      --max_iterations $EPOCHS --seed 1 --checkpoint "$from" \
      agent.params.config.name=$2 agent.params.config.learning_rate=1e-4 > outputs/ft/train_$2.log 2>&1
}

for v in Base FF KpRand; do
  name=ft_$(echo $v | tr 'A-Z' 'a-z')_s1
  train $v $name || { echo "training $name failed (see outputs/ft/train_$name.log)"; exit 1; }
done

for v in base ff kprand; do
  name=ft_${v}_s1
  read ep ck < <(latest $name)
  [ "$ep" -ge "$EPOCHS" ] || { echo "missing final checkpoint for $name"; exit 1; }
  FF=""; [ $v = ff ] && FF="--gravity_ff"
  $RUN $E/${name}.log $E/${name}.npz --controller policyik --checkpoint $ck $FF --task_file $T
  for kp in 0.9 1.1; do
    $RUN $E/${name}_kp${kp}.log $E/${name}_kp${kp}.npz --controller policyik --checkpoint $ck $FF --kp_scale $kp --task_file $T
  done
  if [ $v = kprand ]; then
    for pd in "0.8 1" "1.25 1" "0.95 1" "1.05 1" "1 0.5" "1 2"; do set -- $pd
      $RUN $E/${name}_kp$1_kd$2.log $E/${name}_kp$1_kd$2.npz --controller policyik --checkpoint $ck --kp_scale $1 --kd_scale $2 --task_file $T
    done
  fi
done
echo FT_DONE
