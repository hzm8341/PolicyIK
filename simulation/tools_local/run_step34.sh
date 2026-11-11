#!/usr/bin/env bash
# Steps 3 and 4 in order; each part is idempotent so the GPU guard can stop and restart this script.
cd "$(dirname "$0")/.."
bash tools_local/policyik_franka/run_p2_baselines.sh > outputs/p2/pipeline.log 2>&1; echo "p2 exit $?" >> outputs/step34_status.log
bash tools_local/policyik_ft/run_ft.sh > outputs/ft/pipeline.log 2>&1; echo "ft exit $?" >> outputs/step34_status.log
grep -q P2_BASELINES_DONE outputs/p2/pipeline.log && grep -q FT_DONE outputs/ft/pipeline.log
