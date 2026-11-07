#!/usr/bin/env bash
# GPU job: ROLL unit tests on N XPUs. usage: [TRIES=n] [ONLY="A B C"] run_unit_tests.sh [table|gpu|all]   (table = the 26 GPU-category tests one by one, hangs retried; gpu = multi-GPU files in one pytest; all = full suite)
set -uo pipefail
source /workspace/local/devctl_roll/scripts/env_roll_devctl.sh
MODE=${1:-gpu}; OUT=$R/unit_tests/$(date +%Y%m%d_%H%M%S)_$MODE; mkdir -p $OUT
python /workspace/local/devctl_roll/scripts/check_xpu.py 2>&1 | tail -3
N=$(python -c "import torch; print(torch.xpu.device_count())"); echo "XPUs visible: $N"
[ -e $R/opencl_headers/CL/cl.h ] || python - $R/opencl_headers <<'PYEOF'
import io, shutil, sys, tarfile, urllib.request
d = sys.argv[1]; u = "https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/v2024.10.24.tar.gz"
t = tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(u, timeout=120).read()))
t.extractall("/tmp/oclh"); shutil.copytree("/tmp/oclh/OpenCL-Headers-2024.10.24/CL", d + "/CL", dirs_exist_ok=True)
PYEOF
[ -e $R/opencl_headers/CL/cl.h ] && export CPATH=$R/opencl_headers${CPATH:+:$CPATH}; echo "OpenCL headers: $(ls $R/opencl_headers/CL/cl.h 2>&1)"
[ "$N" -ge 8 ] || echo "WARNING: only $N XPUs visible; the 8-GPU log-prob tests will skip (node gnr17908 exposes 7)"
SHM=$(df -m --output=size /dev/shm | tail -1 | tr -d " "); echo "/dev/shm: ${SHM} MiB"; [ "$SHM" -ge 4096 ] || echo "WARNING: /dev/shm < 4 GiB; vLLM tests need more (devctl test --shm-size=32Gi)"
ray stop >/dev/null 2>&1; ray start --head --port 6392 --include-dashboard=false --disable-usage-stats --temp-dir=$RAY_TMPDIR
cd /workspace/local/ROLL
export MODEL_DOWNLOAD_TYPE=HUGGINGFACE_HUB
DESEL="--deselect tests/models/test_hf_multi_gpus.py::test_hf_multi_gpus_cpu_offload_hf_device_map"
if [ "$MODE" = table ]; then
  TRIES=${TRIES:-3}; SUM=$OUT/summary.tsv; printf "bucket\ttest\tresult\tattempts\thistory\n" > $SUM
  V=tests/third_party/vllm; LP=tests/distributed/strategy/log_probs
  LIST="A|tests/distributed/strategy/context_parallel/test_fsdp2_cp_grad_equivalence.py|900|r
A|tests/models/test_load_generate.py|900|r
A|$V/test_abort.py|1200|r
A|$V/test_model_update.py|1200|r
A|$V/test_vllm_local.py|1200|r
A|$V/test_fp8.py::test_fp8_mem_usage|1200|r
A+D|$V/test_npu_import.py|600|r
B|$LP/test_fsdp_log_probs.py::test_fsdp_log_probs_full|1800|r
B|$LP/test_fsdp_log_probs.py::test_fsdp_log_probs_lora|1800|r
B|$LP/test_fsdp_log_probs.py::test_fsdp_log_probs_cp|1800|r
B|$LP/test_fsdp_log_probs.py::test_fsdp_log_probs_cp_rmpad|1800|r
B|$LP/test_fsdp_vlm_log_probs.py|1800|r
B|$LP/test_fsdp_vlm_layer_states.py|1800|r
B|$LP/test_fsdp_vlm_log_probs_perf.py|1800|r
B|tests/distributed/executor/test_ray_thread_actor_cuda_mem_leak.py|900|1
C|$V/test_fp8.py::test_fp8|900|1
C|$V/test_fp8_perf.py|600|1
E|tests/models/test_hf_multi_gpus.py::test_hf_multi_gpus_cpu_offload_with_hook|900|1"
  i=0
  while IFS="|" read -r b t to rt <&3; do
    [ -n "$t" ] || continue
    [ -z "${ONLY:-}" ] || case " $ONLY " in *" $b "*) ;; *) continue;; esac
    i=$((i+1)); n=$TRIES; [ "$rt" = 1 ] && n=1; hist=""; final=""
    for k in $(seq 1 $n); do
      [ -e $R/STOP ] && { echo "STOP file -> stopping"; break 2; }
      ray stop --force >/dev/null 2>&1; pkill -9 -f "VLLM::EngineCore" 2>/dev/null
      ray start --head --port 6392 --include-dashboard=false --disable-usage-stats --temp-dir=$RAY_TMPDIR >/dev/null 2>&1
      lg=$OUT/$(printf %02d $i)_try$k.log; t0=$SECONDS
      timeout -k 30 $to python -m pytest "$t" -v -rA -o asyncio_mode=auto -p no:cacheprovider --color=no -s > $lg 2>&1; rc=$?
      { [ $rc = 124 ] || [ $rc = 137 ]; } && cp -r $RAY_TMPDIR/session_latest/logs $OUT/$(printf %02d $i)_try${k}_raylogs 2>/dev/null
      ray stop --force >/dev/null 2>&1; pkill -9 -f "VLLM::EngineCore" 2>/dev/null
      c=$(sed "s/\x1b\[[0-9;]*m//g" $lg | grep -E "^(PASSED|FAILED|SKIPPED|ERROR) " | awk '{print $1}' | sort | uniq -c | awk '{printf "%s%s ", $2, $1}')
      if [ $rc = 124 ] || [ $rc = 137 ]; then r="HANG(${to}s)"; elif [ -n "$c" ]; then r="$c"; else r="NO_RESULT(rc=$rc)"; fi
      hist="$hist try$k:${r% } $((SECONDS-t0))s;"; final="${r% }"
      echo "[$(date +%T)] $b $t try $k/$n -> ${r% } ($((SECONDS-t0))s)"
      case "$r" in *FAILED*|*ERROR*|HANG*|NO_RESULT*) ;; *) break;; esac
      [ $k -lt $n ] || break
      for h in 1 2 3; do
        ( timeout 200 python /workspace/local/devctl_roll/scripts/check_xpu.py > $OUT/health_${i}_${k}.log 2>&1; true ) 2>/dev/null
        grep -q "HEALTH OK" $OUT/health_${i}_${k}.log && break; echo "  health check $h/3 not OK, waiting 120 s"; sleep 120
      done
      grep -q "HEALTH OK" $OUT/health_${i}_${k}.log || { echo "GPUs unhealthy after $t -> stopping"; printf "%s\t%s\t%s\t%s\t%s\n" "$b" "$t" "$final" "$k" "$hist" >> $SUM; break 2; }
    done
    printf "%s\t%s\t%s\t%s\t%s\n" "$b" "$t" "$final" "$k" "$hist" >> $SUM
  done 3<<< "$LIST"
  echo "F  tests/models/test_hf_multi_gpus.py::test_hf_multi_gpus_cpu_offload_hf_device_map  NOT RUN (~15 h)"
  echo "===== SUMMARY ($SUM)"; cat $SUM
  echo "UNIT TESTS DONE  logs: $OUT"; exit 0
fi
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
