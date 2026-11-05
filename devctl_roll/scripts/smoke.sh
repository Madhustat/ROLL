#!/usr/bin/env bash
# GPU job: health check, ROLL picks the XPU platform, Ray auto-detects the XPUs, oneCCL version.
set -uo pipefail
source /workspace/local/devctl_roll/scripts/env_roll_devctl.sh
python /workspace/local/devctl_roll/scripts/check_xpu.py 2>&1 | tail -4
python - <<'PY'
import torch, ctypes, os
from roll.platforms import current_platform as p
print("ROLL platform:", type(p).__name__, "| backend:", p.communication_backend, "| supports_tensor_ipc:", p.supports_tensor_ipc, "| xpu count:", torch.xpu.device_count())
for i in range(torch.xpu.device_count()):
    pr = torch.xpu.get_device_properties(i); print(f"  xpu:{i} {pr.name} {pr.total_memory/2**30:.1f} GiB")
import dpctl; print("dpctl level_zero gpus:", dpctl.get_num_devices(backend="level_zero", device_type="gpu"))
print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", "(none)"))
PY
echo "SMOKE DONE"
