#!/usr/bin/env bash
set -euo pipefail
source /workspace/local/devctl_roll/scripts/env_roll_devctl.sh
M=${1:-Qwen/Qwen2.5-0.5B-Instruct}
python -c "from huggingface_hub import snapshot_download as s; print(s('$M', local_dir='$ROLL_WORK/models/$(basename $M)'))"
echo "DOWNLOAD DONE"
