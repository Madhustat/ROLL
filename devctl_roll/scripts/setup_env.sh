#!/usr/bin/env bash
# One-time: build the ROLL XPU venv in /workspace/remote (persists between jobs). Same recipe as the validated bare-metal env.
set -euo pipefail
R=/workspace/remote; L=/workspace/local; E=$L/devctl_roll/env; V=$R/venvs/roll-xpu
command -v python3.12 >/dev/null || { echo "python3.12 not in image"; exit 1; }
python3.12 -c "import venv, ensurepip" 2>/dev/null || { [ "$(id -u)" = 0 ] && apt-get update -qq && apt-get install -y -qq python3.12-venv; }
python3.12 -c "import venv, ensurepip" || { echo "python3.12 venv/ensurepip missing in image"; exit 1; }
[ -e $V ] && { echo "$V exists; delete it to rebuild"; exit 1; }
mkdir -p $R/venvs $R/pip_cache
unset LD_LIBRARY_PATH CCL_ROOT FI_PROVIDER_PATH I_MPI_ROOT VIRTUAL_ENV
python3.12 -m venv $V
$V/bin/pip install -q uv
UV_CACHE_DIR=$R/pip_cache/uv $V/bin/uv pip install --python $V/bin/python \
  -c $E/constraints_xpu_core.txt --override $E/overrides_vllm_xpu_oneapi.txt --override $E/overrides_roll_numpy1.txt \
  'vllm==0.31.0+xpu' -r $E/requirements_roll_xpu.txt \
  -e $L/ROLL/mcore_adapter 'megatron-core>=0.16.0,<0.17.0' \
  --extra-index-url https://wheels.vllm.ai/0.31.0/xpu --extra-index-url https://download.pytorch.org/whl/xpu --index-strategy unsafe-best-match
$V/bin/pip install -q --no-deps dpctl==0.22.1
$V/bin/pip download -q --no-deps oneccl==2022.0.0 -d $R/oneccl_2022_0_0/whl
$V/bin/python -m zipfile -e $R/oneccl_2022_0_0/whl/oneccl-2022.0.0-*.whl $R/oneccl_2022_0_0/x
mkdir -p $R/oneccl_2022_0_0/lib && cp -a $R/oneccl_2022_0_0/x/oneccl-2022.0.0.data/data/lib/libccl.so* $R/oneccl_2022_0_0/lib/
ls -la $R/oneccl_2022_0_0/lib/
$V/bin/python -m pip list --format=freeze > $R/venvs/roll-xpu.pip_freeze.txt
$V/bin/python -c "import torch, vllm, ray, transformers; print('torch', torch.__version__, 'xpu', torch.xpu.is_available(), '| vllm', vllm.__version__, '| ray', ray.__version__, '| transformers', transformers.__version__)"
echo "SETUP DONE: $(wc -l < $R/venvs/roll-xpu.pip_freeze.txt) packages"
