#!/usr/bin/env bash
# GPU job: one ROLL run with vLLM data parallel 2 x tensor parallel 2, local workarounds off.
# usage: run_dp_tp.sh <config_name> [timeout_s]
export ROLL_FSDP2_INIT_WITHOUT_BROADCAST=0 ROLL_LOGITS_CHUNK_SIZE=0
source /workspace/local/devctl_roll/scripts/env_roll_devctl.sh
RD=/workspace/local/ROLL_dp_tp; export PYTHONPATH=$RD ROLL_REPO=$RD
CFG=${1:-moe_dp2_tp2}; TO=${2:-1500}
OUT=$R/dp_tp; mkdir -p $OUT; TAG=${CFG}_$(date +%m%d_%H%M%S); LG=$OUT/$TAG.log
echo "config=$CFG log=$LG ROLL head: $(git -c safe.directory=$RD -C $RD log --oneline -1 2>/dev/null || cat $RD/.git/HEAD)"
python - <<'EOF'
import torch, vllm, ray
from roll.platforms import current_platform as p
n = torch.xpu.device_count()
print("torch", torch.__version__, "| vllm", vllm.__version__, "| ray", ray.__version__, "| platform", type(p).__name__, "| XPUs", n)
for i in range(n):
    print(f"  XPU {i}: uuid={torch.xpu.get_device_properties(i).uuid}")
EOF
N=$(python -c "import torch; print(torch.xpu.device_count())" 2>/dev/null)
NEED=5; case $CFG in *_4xpu) NEED=4;; esac
[ "${N:-0}" -ge $NEED ] || { echo "RESUBMIT: need at least $NEED XPUs, got ${N:-0}"; exit 3; }
[ -f $ROLL_WORK/models/qwen3-moe-tiny-random/config.json ] || { echo "MODEL MISSING: $ROLL_WORK/models/qwen3-moe-tiny-random"; exit 4; }

cat > $OUT/sampler.py <<'EOF'
# Records, per process and per GPU PCI address, the largest GPU memory seen (from /proc/<pid>/fdinfo).
import glob, json, os, sys, time
out, peak = sys.argv[1], {}
while True:
    for d in glob.glob("/proc/[0-9]*"):
        try:
            name = open(d + "/cmdline").read().split("\0")[0][:40]
            if not ("VLLM" in name or name.startswith("ray::")):
                continue
            seen, mem = set(), {}
            for f in glob.glob(d + "/fdinfo/*"):
                kv = dict(l.split(":", 1) for l in open(f).read().splitlines() if ":" in l)
                pdev, cid = kv.get("drm-pdev", "").strip(), kv.get("drm-client-id", "").strip()
                if not pdev or (pdev, cid) in seen:
                    continue
                seen.add((pdev, cid))
                kib = sum(int(v.split()[0]) for k, v in kv.items() if k.startswith("drm-resident-") and v.split())
                mem[pdev] = mem.get(pdev, 0) + kib
            for pdev, kib in mem.items():
                key = f"{name} pid={os.path.basename(d)} | {pdev}"
                peak[key] = max(peak.get(key, 0), kib // 1024)
        except Exception:
            pass
    json.dump(peak, open(out, "w"))
    time.sleep(10)
EOF
python $OUT/sampler.py $OUT/$TAG.gpu_mem.json & SAMPLER=$!
( while true; do echo "== $(date +%T)"; xpu-smi ps 2>/dev/null; sleep 15; done ) > $OUT/$TAG.xpu_ps.txt 2>&1 & PS=$!

ray stop --force >/dev/null 2>&1
ray start --head --port 6392 --include-dashboard=false --disable-usage-stats --temp-dir=$RAY_TMPDIR >/dev/null 2>&1
cd $RD; t0=$SECONDS
timeout -k 30 $TO python examples/start_rlvr_pipeline.py --config_path ../../devctl_roll/dp_tp --config_name $CFG > $LG 2>&1; rc=$?
kill $SAMPLER $PS 2>/dev/null; ray stop --force >/dev/null 2>&1

echo "================ RESULT $CFG ================"
res="FAIL(rc=$rc)"; { [ $rc = 124 ] || [ $rc = 137 ]; } && res="HANG(${TO}s)"
[ $rc = 0 ] && grep -aq "pipeline complete" $LG && res=PASS
echo "result: $res  seconds: $((SECONDS-t0))  steps finished: $(grep -ac 'pipeline step .* finished' $LG)  pipeline complete lines: $(grep -ac 'pipeline complete' $LG)"
echo "--- XPU placement lines (ROLL workers) ---"
grep -aoE "XPU placement: .*" $LG | sort -u | head -20
echo "--- vLLM topology lines ---"
grep -aoE "(data_parallel_rank=[0-9]+|expert parallel enabled[^\"]{0,60}|device_ids[^)]{0,50})" $LG | sort | uniq -c | head -12
echo "--- GPU memory per process (peak MiB, 64 MiB or more) ---"
python - $OUT/$TAG.gpu_mem.json <<'EOF'
import json, sys
try:
    for k, v in sorted(json.load(open(sys.argv[1])).items()):
        if v >= 64:
            print(f"  {v:6d} MiB  {k}")
except Exception as e:
    print("  no sampler data:", e)
EOF
echo "--- xpu-smi ps (last sample with processes; empty if xpu-smi is not in the image) ---"
grep -aE "VLLM|ray::" $OUT/$TAG.xpu_ps.txt | sort -u | tail -12
echo "--- problems ---"
echo "out-of-bounds lines: $(grep -ac 'is out of bounds' $LG)  narrow-mask warnings: $(grep -ac 'narrow device affinity' $LG)  tracebacks: $(grep -ac 'Traceback' $LG)"
grep -aE "Error|is out of bounds" $LG | grep -av "Warning" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-300 | sort -u | head -6
[ "$res" = PASS ] || { echo "--- last 25 log lines ---"; tail -25 $LG | cut -c1-300; }
echo "RUN DONE $CFG $res"
[ "$res" = PASS ]
