# Source inside every devctl job: ROLL on Intel XPU (oneCCL 2022.0.0, proven settings)
R=/workspace/remote; L=/workspace/local
unset CCL_ROOT FI_PROVIDER_PATH I_MPI_ROOT ZE_AFFINITY_MASK ONEAPI_DEVICE_SELECTOR PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
export VIRTUAL_ENV=$R/venvs/roll-xpu
export PATH=$VIRTUAL_ENV/bin:$PATH
for c in /opt/intel/oneapi/compiler/latest/bin /opt/intel/oneapi/compiler/2026.1/bin; do [ -d $c ] && export PATH=$PATH:$c && break; done
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib PYTHONPATH=$L/ROLL
export HF_HOME=$R/hf_home PIP_CACHE_DIR=$R/pip_cache UV_CACHE_DIR=$R/pip_cache/uv KERNELS_CACHE=$R/cache/hf_kernels TMPDIR=$R/tmp
export RAY_TMPDIR=/tmp/ray_roll
export ROLL_WORK=$R/roll_work ROLL_REPO=$L/ROLL
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=8589934592 CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=8589934592
CCLD=$R/oneccl_2022_0_0/lib
[ -e $CCLD/libccl.so.2 ] && export LD_PRELOAD=$CCLD/libccl.so.2:$CCLD/libccl.so.1
mkdir -p $TMPDIR $HF_HOME $ROLL_WORK $RAY_TMPDIR
[ "${ROLL_CCL_IPC:-}" = sockets ] && export CCL_ZE_IPC_EXCHANGE=sockets
