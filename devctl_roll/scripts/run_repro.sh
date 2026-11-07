#!/usr/bin/env bash
# GPU job: run the startup-broadcast reproducer on all visible XPUs, one fresh launch per variant. usage: run_repro.sh [variants...]
source /workspace/local/devctl_roll/scripts/env_roll_devctl.sh
N=$(python -c "import torch; print(torch.xpu.device_count())"); OUT=$R/unit_tests/repro_$(date +%H%M%S); mkdir -p $OUT
VARIANTS=${*:-roll warmup setdev algo}
echo "XPUs: $N ($(python -c "import torch; print([torch.xpu.get_device_name(i)[-12:] for i in range(torch.xpu.device_count())])"))  variants: $VARIANTS  logs: $OUT"
for v in $VARIANTS; do
  P=$((29700 + RANDOM % 200)); X=""; case $v in *algo*) X="CCL_BCAST=ring";; esac; U=""; case $v in *nothresh*) U="-u CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD -u CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD";; esac
  for r in $(seq 0 $((N-1))); do env $U ZE_AFFINITY_MASK=$r $X timeout 240 python /workspace/local/devctl_roll/scripts/repro_bcast.py $r $N $P $v > $OUT/${v}_r$r.log 2>&1 & done; wait
  ok=$(grep -l BROADCAST_OK $OUT/${v}_r*.log 2>/dev/null | wc -l)
  echo "RESULT variant=$v ${X:+($X) }ranks_ok=$ok/$N $( [ $ok = $N ] && echo PASS || echo HANG_OR_FAIL ) $(grep -h 'data_ok=False' $OUT/${v}_r*.log | head -1)"
  [ $ok = $N ] || { grep -hE 'Error|error' $OUT/${v}_r*.log | sort | uniq -c | head -3; sleep 15; }
done
echo "REPRO DONE"
