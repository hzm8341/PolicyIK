#!/usr/bin/env bash
# P1 tuning grid on the tuning set (never on eval_tasks). Selection: strict success on the tuning set.
cd "$(dirname "$0")/.."
T=outputs/p1/tune_tasks.npz
R=outputs/p1/tune
tools_local/run_eval.sh $R/policyik_s1.log $R/policyik_s1.npz --controller policyik --checkpoint checkpoints/policyik_s1.pth --task_file $T
for c in dls qp; do
  for k in 0.5 1 1.5 2 3 5; do
    for lead in 0.05 0.1 0.15 0.25 0.5 10; do
      tools_local/run_eval.sh $R/${c}_k${k}_lead${lead}.log $R/${c}_k${k}_lead${lead}.npz --controller $c --k $k --lead $lead --task_file $T
    done
  done
done
echo TUNING_DONE
