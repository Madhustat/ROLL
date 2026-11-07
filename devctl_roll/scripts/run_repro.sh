#!/usr/bin/env bash
# GPU job: run the startup-broadcast reproducer on all visible XPUs, REPEAT fresh launches per variant. usage: [REPEAT=n] [DEVS="0 1 .."] run_repro.sh [variants...]
source /workspace/local/devctl_roll/scripts/env_roll_devctl.sh
N=$(python -c "import torch; print(torch.xpu.device_count())"); OUT=$R/unit_tests/repro_$(date +%H%M%S); mkdir -p $OUT
VARIANTS=${*:-roll warmup setdev algo}
ALL=$N; DEVS=${DEVS:-$(seq -s " " 0 $((N-1)))}; DA=($DEVS); N=${#DA[@]}
echo "kernel $(uname -r); driver pkgs: $(dpkg -l 2>/dev/null | awk '/intel-opencl-icd|libze-intel-gpu1|libze1|intel-level-zero-gpu/{printf "%s=%s ", $2, $3}')"
python -c "import torch; [print('  xpu:%d driver=%s mem=%.1fGiB' % (i, torch.xpu.get_device_properties(i).driver_version, torch.xpu.get_device_properties(i).total_memory/2**30)) for i in range(torch.xpu.device_count())]" 2>/dev/null | head -2
for c in /sys/class/drm/card[0-9]*; do [ -e $c/device/numa_node ] && echo "  $(basename $c) pci=$(basename $(readlink -f $c/device)) numa=$(cat $c/device/numa_node)"; done
echo "DEVS used: $DEVS (of $ALL)"
REPEAT=${REPEAT:-1}
echo "REPEAT=$REPEAT  XPUs: $N ($(python -c "import torch; print([torch.xpu.get_device_name(i)[-12:] for i in range(torch.xpu.device_count())])"))  variants: $VARIANTS  logs: $OUT"
for v in $VARIANTS; do
 pass=0
 for k in $(seq 1 $REPEAT); do
  P=$((29700 + RANDOM % 200)); X=""; case $v in *algo*) X="CCL_BCAST=ring";; esac; U=""; case $v in *nothresh*) U="-u CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD -u CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD";; esac
  for r in $(seq 0 $((N-1))); do ( env $U ZE_AFFINITY_MASK=${DA[$r]} $X timeout 240 python /workspace/local/devctl_roll/scripts/repro_bcast.py $r $N $P $v; echo "EXIT rc=$?" ) > $OUT/${v}_${k}_r$r.log 2>&1 & done; wait
  ok=$(grep -l BROADCAST_OK $OUT/${v}_${k}_r*.log 2>/dev/null | wc -l)
  [ $ok = $N ] && pass=$((pass+1))
  echo "  launch $k/$REPEAT variant=$v ${X:+($X) }ranks_ok=$ok/$N $( [ $ok = $N ] && echo PASS || echo HANG_OR_FAIL ) $(grep -h 'data_ok=False' $OUT/${v}_${k}_r*.log | head -1)"
  [ $ok = $N ] || { for f in $OUT/${v}_${k}_r*.log; do echo "    $(basename $f .log): last=$(grep -o 'STAGE [a-z_]*' $f | tail -1) $(grep -o 'EXIT rc=[0-9]*' $f) frame=$(grep -m1 -A1 'most recent call first' $f | tail -1 | sed 's/^ *//')"; done; grep -hE 'Error|error' $OUT/${v}_${k}_r*.log | sort | uniq -c | head -3; sleep 15; }
 done
 echo "RESULT variant=$v ${X:+($X) }passed $pass of $REPEAT launches"
done
echo "REPRO DONE"
