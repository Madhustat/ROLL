#!/usr/bin/env bash
# GPU job: run one ROLL launcher with a config from devctl_roll/configs. usage: run_roll.sh <launcher.py> <config_name>
set -uo pipefail
source /workspace/local/devctl_roll/scripts/env_roll_devctl.sh
LAUNCHER=${1:-start_rlvr_pipeline.py}; CFG=${2:-grpo_qwen25_05b_1xpu_colocated}
ray stop >/dev/null 2>&1
ray start --head --port 6392 --include-dashboard=false --disable-usage-stats --temp-dir=$RAY_TMPDIR
python -c "import ray; ray.init(address='auto'); print('Ray resources:', ray.cluster_resources())"
cd /workspace/local/ROLL
python examples/$LAUNCHER --config_path ../../devctl_roll/configs --config_name $CFG
rc=$?
ray stop >/dev/null 2>&1
echo "RUN DONE rc=$rc"
exit $rc
