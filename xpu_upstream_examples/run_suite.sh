#!/usr/bin/env bash
# ROLL upstream example pipelines on 2 Intel XPUs, one at a time. Before each example: wait for free GPUs, clean caches /
# checkpoints / outputs / temp, health check. During: watchdog (time limit, stalled log, xe kernel fault). After: kill
# leftovers, 10 min cool-down if a GPU fault appeared, health check. Results: $WORK/results.tsv, logs: $WORK/runs/<name>/.
# Usage: ./run_suite.sh [example ...]   (no arguments = every example in examples.list). Create $WORK/STOP to stop.
# Runs in the background by default (output: $WORK/suite.out); SUITE_FG=1 keeps it in the foreground. RESUME=1 skips
# examples whose latest result is PASS. One suite at a time; leftovers of a killed earlier run are removed first.
# RLVR PASS = "pipeline complete", rc 0 and every step finished ("pipeline step N finished") or skipped by ROLL itself
# ("Current step will be skipped": all responses masked by max_len_mask/difficulty_mask; shown as PASS(...)).
# Settings (environment): ROLL_DIR VENV ONECCL_LIB MODELS_DIR DATA_DIR WORK MIN_RAM_GB (stop an example when available
# host RAM drops below it, default 10)
set -u
SUITE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$SUITE/work}
if [ -z "${SUITE_DETACHED:-}" ] && [ -z "${SUITE_FG:-}" ]; then
  mkdir -p $WORK; SUITE_DETACHED=1 WORK=$WORK setsid nohup "$SUITE/run_suite.sh" "$@" >> $WORK/suite.out 2>&1 < /dev/null &
  echo "suite started in the background (pid $!); follow: tail -f $WORK/suite.out; results: $WORK/results.tsv; stop: touch $WORK/STOP"
  exit 0
fi
LOCK=/tmp/roll_suite_$USER.pid
OTHER=$(ps -eo pid=,ppid=,args= | awk -v me=$$ -v pp=$PPID '$3 ~ /bash$/ && $4 ~ /run_suite\.sh$/ && $1 != me && $2 != me && $1 != pp {print $1}')
[ -n "$OTHER" ] && { echo "another suite is running (pid $OTHER), not starting"; exit 1; }
echo $$ > $LOCK
ROLL_DIR=${ROLL_DIR:-$(cd "$SUITE/.." && pwd)}
VENV=${VENV:-/madhu_Roll/stage1_20261102/venv}
ONECCL_LIB=${ONECCL_LIB:-/madhu_Roll/oneccl_2022_0_0/x/oneccl-2022.0.0.data/data/lib}
MODELS_DIR=${MODELS_DIR:-/madhu_Roll/stage1_20261102/roll_work/models}
DATA_DIR=${DATA_DIR:-/madhu_Roll/stage1_20261102/roll_work/data}
RAY_TMP=/tmp/ray_roll_suite_$USER RAY_PORT=6397 MIN_FREE_GB=40 MIN_RAM_GB=${MIN_RAM_GB:-10}
SUITE_TMP=${SUITE_TMP:-/tmp/roll_suite_tmp_$USER}
mkdir -p $WORK/runs $WORK/configs $WORK/out $WORK/cache $WORK/hf_home $SUITE_TMP
RES=$WORK/results.tsv; [ -s $RES ] || printf "time\texample\tresult\trc\tsteps\tseconds\tgpu_faults\thealth\tplacement\terror\n" > $RES

KERNELS_SEED=${KERNELS_SEED:-/madhu_Roll/xpu_explicit_device_placement_20261111/cache/hf_kernels}
[ -n "$(ls -A $WORK/cache/hf_kernels 2>/dev/null)" ] || { mkdir -p $WORK/cache/hf_kernels; cp -r $KERNELS_SEED/. $WORK/cache/hf_kernels/ 2>/dev/null; }
for f in $SUITE/configs/*.yaml; do
  sed -e "s#__MODELS_DIR__#$MODELS_DIR#g" -e "s#__DATA_DIR__#$DATA_DIR#g" -e "s#__OUT_DIR__#$WORK/out#g" $f > $WORK/configs/$(basename $f)
done
CREL=$(python3 -c "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" $WORK/configs $ROLL_DIR/examples)

ENVV=(env -i HOME="$HOME" USER="$USER" PATH="$VENV/bin:/opt/intel/oneapi/compiler/2026.1/bin:/usr/bin:/bin" VIRTUAL_ENV=$VENV
  LD_LIBRARY_PATH=$VENV/lib LD_PRELOAD=$ONECCL_LIB/libccl.so.2:$ONECCL_LIB/libccl.so.1 PYTHONPATH=$ROLL_DIR
  PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=$SUITE_TMP HF_HOME=$WORK/hf_home HF_HUB_OFFLINE=1
  XDG_CACHE_HOME=$WORK/cache/xdg TRITON_CACHE_DIR=$WORK/cache/triton TORCHINDUCTOR_CACHE_DIR=$WORK/cache/inductor
  KERNELS_CACHE=$WORK/cache/hf_kernels CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=8589934592
  CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=8589934592 RAY_TMPDIR=$RAY_TMP RAY_ADDRESS=127.0.0.1:$RAY_PORT RAY_DEDUP_LOGS=0)

clean() { rm -rf $WORK/out/* $WORK/cache/xdg $WORK/cache/triton $WORK/cache/inductor $SUITE_TMP/* $RAY_TMP; mkdir -p $SUITE_TMP; }
health() { "${ENVV[@]}" timeout 150 python $SUITE/health_check.py 2>&1 | grep -q "HEALTH OK"; }
gpu_busy() { local p; for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q /dev/dri && { echo 1; return; }; done; echo 0; }
ours() { grep -l "SUITE_MARKER=$1" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3; }
kill_ours() { local p; for p in $(ours "$1"); do kill -TERM $p 2>/dev/null; done; sleep 10
  for p in $(ours "$1"); do kill -9 $p 2>/dev/null; done; "${ENVV[@]}" ray stop --force >/dev/null 2>&1; sleep 5; }
CUR=""
on_signal() { echo "$(date +%T) suite stopped by signal${CUR:+ during $CUR}, cleaning up"; [ -n "$CUR" ] && kill_ours $CUR; clean; rm -f $LOCK; exit 1; }
trap on_signal TERM INT HUP
trap 'rm -f $LOCK' EXIT
faults_since() { sudo -n dmesg 2>/dev/null | python3 -c "
import sys,re,time
up=float(open('/proc/uptime').read().split()[0]); boot=time.time()-up; t0=float(sys.argv[1])
x=[time.strftime('%H:%M:%S',time.localtime(boot+float(m.group(1))))+' '+l[l.find('xe 0000')+3:l.find('xe 0000')+15]+' '+l[l.find('GT0:')+5:].strip()[:40]
   for l in sys.stdin if (m:=re.match(r'\[\s*(\d+\.\d+)\]',l)) and boot+float(m.group(1))>=t0 and 'xe 0000' in l and re.search('reset|CAT|Fault resp|wedged|timedout',l)]
print(' | '.join(dict.fromkeys(x)) or 'none')" "$1"; }

summarize() {  # summarize <run dir> <result> <stop reason>: keep only what validates the result, a few KB
  local O=$1 L=$1/run.log S=$1/summary.txt
  { echo "result: $2 (stopped: $3)"; cat $O/meta.txt
    echo "--- placement (worker: physical XPU) ---"
    sed 's/\x1b\[[0-9;]*m//g' $L | grep -oE "\[[A-Za-z_-]+-[0-9]+-G[0-9]+ [0-9]+ / [0-9]+\]\[PID [0-9]+\] XPU placement: .*" | sort -u
    echo "--- steps ---"; python3 $SUITE/rlvr_metrics.py $L 2>/dev/null | tail -24
    grep -c "pipeline complete" $L | sed 's/^/pipeline complete lines: /'
    echo "--- peak GPU memory per worker (GB, from ROLL's own log) ---"
    sed 's/\x1b\[[0-9;]*m//g' $L | grep -E "_end_onload, memory allocated" \
      | sed -E 's/.*\[([A-Za-z_-]+-[0-9]+-G[0-9]+) .*\] ([A-Za-z_/]+), .*max reserved \(GB\): ([0-9.]+).*/\1 \2 max_reserved=\3/' | sort -u | tail -12
    echo "--- errors ---"; sed 's/\x1b\[[0-9;]*m//g' $L | grep -E "Error|Exception|Traceback|SIGSEGV|SIGABRT" | grep -vE "pynvml|UserWarning|triton_bundler" | sort -u | head -30
    echo "--- last 40 lines (noise removed) ---"
    sed 's/\x1b\[[0-9;]*m//g' $L | grep -vE "file_system_monitor|shm_broadcast|perf_utils|Per-operation|Time range|_DATA|RemoteBatch|remote_protocol|^\s*$" | tail -40 | cut -c1-300
  } > $S 2>/dev/null
}

run_one() {  # run_one <name> <launcher> <config> <kind> <steps> <limit> <stall>
  local name=$1 L=$2 C=$3 kind=$4 exp=$5 LIMIT=$6 STALL=$7 O=$WORK/runs/$1
  [ -f $WORK/STOP ] && { echo "STOP file found"; exit 0; }
  local avail=$(df --output=avail -BG $WORK | tail -1 | tr -dc 0-9)
  [ "$avail" -lt $MIN_FREE_GB ] && { echo "only ${avail} GB free before $name, stopping"; exit 1; }
  CUR=$name; local free=0; while [ $free -lt 60 ]; do [ "$(gpu_busy)" = 0 ] && free=$((free + 10)) || free=0; sleep 10; done
  clean; rm -rf $O; mkdir -p $O
  health || { sleep 300; health || { echo "GPUs unhealthy before $name, stopping"; exit 1; }; }
  local t0=$(date +%s)
  "${ENVV[@]}" SUITE_MARKER=$name bash -c "ray stop --force >/dev/null 2>&1; ray start --head --port $RAY_PORT --num-gpus 2 \
    --include-dashboard=false --disable-usage-stats --temp-dir $RAY_TMP >/dev/null 2>&1"
  echo "ROLL $(git -C $ROLL_DIR rev-parse --short HEAD) dirty=[$(git -C $ROLL_DIR status --porcelain | grep -v '^??' | tr '\n' ' ')] config=$C" > $O/meta.txt
  ( cd $ROLL_DIR && exec setsid "${ENVV[@]}" SUITE_MARKER=$name python examples/$L --config_path $CREL --config_name $C ) > $O/run.log 2>&1 &
  local PID=$! last=0 same=0 reason=exited rc=NA peak_low=999
  while kill -0 $PID 2>/dev/null; do
    sleep 5; local sz=$(stat -c %s $O/run.log)
    [ "$sz" = "$last" ] && same=$((same + 5)) || { same=0; last=$sz; }
    [ "$(faults_since $t0)" != none ] && { reason="GPU fault"; break; }
    local ram=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); [ $ram -gt ${peak_low:-999} ] || peak_low=$ram
    [ $ram -lt $MIN_RAM_GB ] && { reason="host RAM available ${ram} GB < ${MIN_RAM_GB} GB"; break; }
    [ $same -ge $STALL ] && { reason="stalled ${same}s"; break; }
    [ $(( $(date +%s) - t0 )) -ge $LIMIT ] && { reason="time limit"; break; }
  done
  kill -0 $PID 2>/dev/null || { wait $PID; rc=$?; }
  local secs=$(( $(date +%s) - t0 ))
  kill_ours $name
  local err=$(grep -E "^[A-Za-z.]*(Error|Exception): |RayTaskError" $O/run.log | sed 's/\x1b\[[0-9;]*m//g' | head -1 | cut -c1-160 | tr '\t' ' ')
  local st=FAIL steps=NA
  if [ $kind = rlvr ]; then
    local m=$(python3 $SUITE/rlvr_metrics.py $O/run.log 2>/dev/null)
    steps=$(echo "$m" | sed -n 's/steps completed: //p'); steps=${steps:-0}
    local skipped=$(echo "$m" | sed -n 's/steps skipped by ROLL.*: //p')
    grep -q "pipeline complete" $O/run.log && [ "$rc" = 0 ] && [ "$steps" -ge "$exp" ] && st=PASS
    [ $st = PASS ] && [ "${skipped:-0}" -gt 0 ] && st="PASS($skipped step skipped by ROLL: all responses masked)"
  else
    [ "$rc" = 0 ] && [ -z "$err" ] && st=PASS
  fi
  [ $st = FAIL ] && [ "$reason" != exited ] && st="FAIL($reason)"
  local place=$(sed 's/\x1b\[[0-9;]*m//g' $O/run.log | grep -oE "\[[A-Za-z_-]+-[0-9]+-G[0-9]+ [0-9]+ / [0-9]+\]\[PID [0-9]+\] XPU placement: .*current_device=[0-9]+" \
    | sed -E 's/^\[([A-Za-z_-]+-[0-9]+-G[0-9]+) .*current_device=([0-9]+)/\1:xpu\2/' | sort -u | tr '\n' ' ')
  echo "lowest host RAM available during the run: ${peak_low} GB" >> $O/meta.txt
  summarize $O "$st" "$reason"; [ "$st" = PASS ] && rm -f $O/run.log || gzip -f $O/run.log
  sleep 30; local f=$(faults_since $t0); [ "$f" != none ] && sleep 600
  local h; health && h=OK || { sleep 300; health && h=OK || h=FAILED; }
  clean
  printf "%s\t%s\t%s\t%s\t%s/%s\t%s\t%s\t%s\t%s\t%s\n" "$(date '+%m-%d %H:%M')" $name "$st" $rc $steps $exp $secs "$f" $h "$place" "$err" | tee -a $RES
  CUR=""
  [ $h = FAILED ] && { echo "GPUs unhealthy after $name, stopping"; exit 1; }
}

for p in $(grep -l "SUITE_MARKER=" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3); do
  echo "$(date +%T) removing leftover from an earlier run: pid $p $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-80)"
  kill -TERM $p 2>/dev/null
done
sleep 5; for p in $(grep -l "SUITE_MARKER=" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3); do kill -9 $p 2>/dev/null; done
"${ENVV[@]}" ray stop --force >/dev/null 2>&1; clean

while IFS='|' read -r name L C kind exp lim stall; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  [ $# -gt 0 ] && ! printf ' %s ' "$@" | grep -q " $name " && continue
  [ -n "${RESUME:-}" ] && [[ "$(awk -F'\t' -v n=$name '$2 == n {r = $3} END {print r}' $RES)" == PASS* ]] && { echo "$name already PASS, skipped"; continue; }
  run_one $name $L $C $kind $exp $lim $stall
done < $SUITE/examples.list
echo "SUITE DONE (latest result per example): $(awk -F'\t' 'NR > 1 {r[$2] = $3} END {for (n in r) print r[n]}' $RES | sed 's/(.*//' | sort | uniq -c | tr '\n' ' ')"
