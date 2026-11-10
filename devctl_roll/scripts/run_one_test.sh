#!/usr/bin/env bash
# GPU job: one pytest node id in a fresh pod, upstream ROLL code paths (local workarounds off). usage: [REQUIRE_XPUS=n] run_one_test.sh <test_id> [timeout_s]
export ROLL_FSDP2_INIT_WITHOUT_BROADCAST=0 ROLL_LOGITS_CHUNK_SIZE=0
source /workspace/local/devctl_roll/scripts/env_roll_devctl.sh
T=$1; TO=${2:-1800}
N=$(python -c "import torch; print(torch.xpu.device_count())" 2>/dev/null)
[ "$N" -ge "${REQUIRE_XPUS:-0}" ] || { echo "RESUBMIT: need ${REQUIRE_XPUS} XPUs, got $N"; exit 3; }
[ -e $R/opencl_headers/CL/cl.h ] && export CPATH=$R/opencl_headers${CPATH:+:$CPATH}
OUT=$R/unit_tests/one; mkdir -p $OUT; LG=$OUT/$(echo "$T" | tr '/:' '__')_$(date +%m%d_%H%M%S).log
echo "test=$T XPUs=$N workarounds: FSDP2_INIT_WITHOUT_BROADCAST=$ROLL_FSDP2_INIT_WITHOUT_BROADCAST LOGITS_CHUNK_SIZE=$ROLL_LOGITS_CHUNK_SIZE log=$LG"
grep -h "^prompt_length" /workspace/local/ROLL/tests/distributed/strategy/log_probs/log_probs_fsdp_cp*_config.yaml | sort -u | sed 's/^/cp configs: /'
ray stop --force >/dev/null 2>&1; ray start --head --port 6392 --include-dashboard=false --disable-usage-stats --temp-dir=$RAY_TMPDIR >/dev/null 2>&1
cd /workspace/local/ROLL; export MODEL_DOWNLOAD_TYPE=HUGGINGFACE_HUB; t0=$SECONDS
timeout -k 30 $TO python -m pytest "$T" -v -rA -p no:cacheprovider --color=no -s > $LG 2>&1; rc=$?
ray stop --force >/dev/null 2>&1
c=$(sed "s/\x1b\[[0-9;]*m//g" $LG | grep -E "^(PASSED|FAILED|SKIPPED|ERROR) " | awk '{print $1}' | sort | uniq -c | awk '{printf "%s%s ", $2, $1}')
{ [ $rc = 124 ] || [ $rc = 137 ]; } && c="HANG(${TO}s)"
echo "ONE_RESULT ${c:-NO_RESULT(rc=$rc)} $((SECONDS-t0))s test=$T"
sed "s/\x1b\[[0-9;]*m//g" $LG | grep -E "OutOfMemory|^E  .*Error" | sort -u | head -3
