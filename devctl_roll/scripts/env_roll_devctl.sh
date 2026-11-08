# Source inside every devctl job: ROLL on Intel XPU (oneCCL 2022.0.0, proven settings)
R=/workspace/remote; L=/workspace/local
unset CCL_ROOT FI_PROVIDER_PATH I_MPI_ROOT ZE_AFFINITY_MASK ONEAPI_DEVICE_SELECTOR PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
export VIRTUAL_ENV=$R/venvs/roll-xpu
export PATH=$VIRTUAL_ENV/bin:$PATH
for c in /opt/intel/oneapi/compiler/latest/bin /opt/intel/oneapi/compiler/2026.1/bin; do [ -d $c ] && export PATH=$PATH:$c && break; done
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib PYTHONPATH=$L/ROLL
if [ -n "${NEO:-}" ] && [ "$NEO" != none ] && [ -d $R/neo_$NEO/debs ] && [ ! -e /tmp/.neo_$NEO ]; then dpkg -i $R/neo_$NEO/debs/*.deb > /tmp/neo_install.log 2>&1 && ldconfig && touch /tmp/.neo_$NEO || echo "NEO $NEO install FAILED: $(tail -3 /tmp/neo_install.log)"; fi
export HF_HOME=$R/hf_home PIP_CACHE_DIR=$R/pip_cache UV_CACHE_DIR=$R/pip_cache/uv KERNELS_CACHE=$R/cache/hf_kernels TMPDIR=$R/tmp
export RAY_TMPDIR=/tmp/ray_roll
export ROLL_WORK=$R/roll_work ROLL_REPO=$L/ROLL
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=8589934592 CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=8589934592
export ROLL_FSDP2_INIT_WITHOUT_BROADCAST=${ROLL_FSDP2_INIT_WITHOUT_BROADCAST:-1} ROLL_LOGITS_CHUNK_SIZE=${ROLL_LOGITS_CHUNK_SIZE:-128}
CCLD=$R/oneccl_2022_0_0/lib
[ -e $CCLD/libccl.so.2 ] && export LD_PRELOAD=$CCLD/libccl.so.2:$CCLD/libccl.so.1
mkdir -p $TMPDIR $HF_HOME $ROLL_WORK $RAY_TMPDIR
[ -e $R/opencl_headers/CL/cl.h ] && export CPATH=$R/opencl_headers${CPATH:+:$CPATH}
[ "${ROLL_CCL_IPC:-}" = sockets ] && export CCL_ZE_IPC_EXCHANGE=sockets
true
