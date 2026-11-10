#!/usr/bin/env bash
# Local 2xB70: one clean run of a log-prob test (upstream code paths, no workarounds). usage: run_local.sh <test_id> <tag>
S=/madhu_Roll/stage1_20261102; O=$S/clean2xpu; T=$1; TAG=$2; LG=$O/${TAG}.log
source $S/stage1_env.sh >/dev/null 2>&1
CCLD=/madhu_Roll/oneccl_2022_0_0/x/oneccl-2022.0.0.data/data/lib
export LD_PRELOAD=$CCLD/libccl.so.2:$CCLD/libccl.so.1 ROLL_FSDP2_INIT_WITHOUT_BROADCAST=0 ROLL_LOGITS_CHUNK_SIZE=0 MODEL_DOWNLOAD_TYPE=HUGGINGFACE_HUB
cd $S/ROLL
J=$(echo "$T" | sed 's/.*::test_fsdp_log_probs_/test_fsdp_log_probs_/').json; rm -f "$J"
export RAY_TMPDIR=/tmp/ray_clean RAY_ADDRESS=127.0.0.1:6392
ray stop --force >/dev/null 2>&1; rm -rf $RAY_TMPDIR; ray start --head --port 6392 --num-gpus 2 --include-dashboard=false --disable-usage-stats --temp-dir $RAY_TMPDIR >/dev/null 2>&1
G=$(python -c "import ray; ray.init(address='auto', log_to_driver=False); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null)
echo "PLACEMENT ray_GPU=$G ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-unset} ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-unset} devices=$(python -c 'import torch; print([torch.xpu.get_device_name(i)[-12:] for i in range(torch.xpu.device_count())])')"
[ "$G" = 2 ] || { echo "ABORT: Ray sees $G GPUs, need 2"; ray stop --force >/dev/null 2>&1; exit 2; }
echo "START $(date +%T) test=$T xpus=$(python -c 'import torch; print(torch.xpu.device_count())') workarounds=$ROLL_FSDP2_INIT_WITHOUT_BROADCAST/$ROLL_LOGITS_CHUNK_SIZE oneccl=$(basename $(readlink -f $CCLD/libccl.so.2))"
t0=$SECONDS; timeout -k 30 900 python -m pytest "$T" -v -rA -p no:cacheprovider --color=no -s > $LG 2>&1; rc=$?; dur=$((SECONDS-t0))
ray stop --force >/dev/null 2>&1
res=$(sed "s/\x1b\[[0-9;]*m//g" $LG | grep -oE "^(PASSED|FAILED|SKIPPED|ERROR) " | sort | uniq -c | awk '{printf "%s%s ", $2, $1}'); { [ $rc = 124 ] || [ $rc = 137 ]; } && res="HANG(900s)"
mem=$(sed "s/\x1b\[[0-9;]*m//g" $LG | grep -oE "memory max reserved \(GB\): [0-9.]+" | awk '{print $NF}' | sort -g | tail -1)
init=$(grep -c " initialized" $LG)
diff=$(python - "$J" <<'EOF' 2>/dev/null
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"rows={len(rows)} max_diff_max={max(r.get('diff_max', 0) for r in rows):.3e} max_diff_mean={max(r.get('diff_mean', 0) for r in rows):.3e}")
EOF
)
echo "RESULT tag=$TAG ${res:-NO_RESULT(rc=$rc)} ${dur}s workers_initialized=$init peak_reserved_GB=${mem:-?} ${diff:-no_json}"
