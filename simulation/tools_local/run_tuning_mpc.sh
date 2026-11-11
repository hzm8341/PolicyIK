#!/usr/bin/env bash
cd "$(dirname "$0")/.."
T=outputs/p1/tune_tasks.npz
R=outputs/p1/tune
for tr in 0.1 10; do
  for k in 1 1.5 2 3; do
    for lead in 0.1 0.15 0.25; do
      tools_local/run_eval.sh $R/mpc_k${k}_lead${lead}_tr${tr}.log $R/mpc_k${k}_lead${lead}_tr${tr}.npz --controller mpc --k $k --lead $lead --trust $tr --task_file $T
    done
  done
done
echo MPC_TUNING_DONE
