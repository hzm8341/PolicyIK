#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q TUNING_DONE outputs/p1/tuning_run.log 2>/dev/null; do sleep 30; done
tools_local/run_tuning_mpc.sh > outputs/p1/tuning_mpc_run.log 2>&1
source env_local.sh
PYTHONPATH=$PWD:$PYTHONPATH python tools_local/summarize_tuning.py outputs/p1/tune > outputs/p1/tune/tuning_summary.txt 2>&1
tools_local/run_p1_main.sh > outputs/p1/main_run.log 2>&1
echo PIPELINE_DONE
