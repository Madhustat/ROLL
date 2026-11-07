#!/usr/bin/env bash
# GPU job: ROLL unit tests on N XPUs. usage: run_unit_tests.sh [gpu|all]   (gpu = the multi-GPU tests that skipped on 2 XPUs; all = full suite)
set -uo pipefail
source /workspace/local/devctl_roll/scripts/env_roll_devctl.sh
MODE=${1:-gpu}; OUT=$R/unit_tests/$(date +%Y%m%d_%H%M%S)_$MODE; mkdir -p $OUT
python /workspace/local/devctl_roll/scripts/check_xpu.py 2>&1 | tail -3
N=$(python -c "import torch; print(torch.xpu.device_count())"); echo "XPUs visible: $N"
ray stop >/dev/null 2>&1; ray start --head --port 6392 --include-dashboard=false --disable-usage-stats --temp-dir=$RAY_TMPDIR
cd /workspace/local/ROLL
export MODEL_DOWNLOAD_TYPE=HUGGINGFACE_HUB
DESEL="--deselect tests/models/test_hf_multi_gpus.py::test_hf_multi_gpus_cpu_offload_hf_device_map"
if [ "$MODE" = gpu ]; then
  T="tests/distributed/strategy/log_probs tests/distributed/strategy/context_parallel/test_fsdp2_cp_grad_equivalence.py tests/models/test_hf_multi_gpus.py tests/third_party/vllm/test_vllm_local.py tests/third_party/vllm/test_model_update.py tests/third_party/vllm/test_abort.py tests/distributed/executor/test_ray_thread_actor_cuda_mem_leak.py"
else
  T="tests"
fi
python -m pytest $T $DESEL -v -rA -o asyncio_mode=auto --junitxml=$OUT/junit.xml 2>&1 | tee $OUT/pytest.log
rc=${PIPESTATUS[0]}
ray stop >/dev/null 2>&1
grep -E "^(PASSED|FAILED|SKIPPED|ERROR)" $OUT/pytest.log | awk '{print $1}' | sort | uniq -c
echo "UNIT TESTS DONE rc=$rc  log: $OUT/pytest.log"
exit $rc
