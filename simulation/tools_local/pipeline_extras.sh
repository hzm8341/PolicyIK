#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q PIPELINE_DONE outputs/p1/pipeline.log 2>/dev/null; do sleep 60; done
tools_local/run_p1_robust.sh > outputs/p1/robust_run.log 2>&1
echo EXTRAS_DONE
