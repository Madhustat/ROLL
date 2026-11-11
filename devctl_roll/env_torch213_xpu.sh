# Isolated test env: same stack as env_roll_xpu.sh except torch 2.13.0+xpu instead of 2.14.0+xpu.
# Built fresh in venv_torch213_xpu; does not touch venv_roll_xpu. Usage: source env_torch213_xpu.sh
unset CCL_ROOT FI_PROVIDER_PATH I_MPI_ROOT ZE_AFFINITY_MASK ONEAPI_DEVICE_SELECTOR
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
[[ -n ${VIRTUAL_ENV:-} ]] && deactivate 2>/dev/null
V=/madhu_Roll/stage1_20261102/venv_torch213_xpu
CCLD=/madhu_Roll/oneccl_2022_0_0/x/oneccl-2022.0.0.data/data/lib   # same oneCCL 2022.0.0 build used for the torch-2.14 hang
export LD_LIBRARY_PATH=$V/lib
export LD_PRELOAD=$CCLD/libccl.so.2:$CCLD/libccl.so.1
source "$V/bin/activate"
export PATH="$PATH:/opt/intel/oneapi/compiler/2026.1/bin"
