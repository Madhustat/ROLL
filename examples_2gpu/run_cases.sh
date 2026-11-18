#!/usr/bin/env bash
# Run the 2-GPU cases in cases.list, one at a time, on whatever accelerator ROLL detects.
# usage: [ROLL_DIR=<checkout>] [WORK=<dir>] ./run_cases.sh [case ...]   (no arguments = every case)
# ROLL_DIR defaults to the checkout this folder is in; point it at another checkout (e.g. upstream main) to compare.
# Results: $WORK/results_<checkout name>.tsv; logs: $WORK/logs/<checkout name>/<case>.log. Stop: touch $WORK/STOP
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROLL_DIR=$(cd "${ROLL_DIR:-$HERE/..}" && pwd)
WORK=${WORK:-$HERE/work}
TAG=$(basename "$ROLL_DIR")
export ROLL_WORK=$WORK ROLL_REPO=$ROLL_DIR
LOGS=$WORK/logs/$TAG
RES=$WORK/results_$TAG.tsv
mkdir -p "$WORK/configs" "$LOGS"
ln -sfn "$ROLL_DIR/examples/config" "$WORK/config"
for f in "$HERE"/configs/*.yaml; do
  sed -e "s#__MODELS_DIR__#$WORK/models#g" -e "s#__DATA_DIR__#$WORK/data#g" -e "s#__OUT_DIR__#$WORK/out#g" \
    "$f" > "$WORK/configs/$(basename "$f")"
done
CREL=$(python3 -c "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "$WORK/configs" "$ROLL_DIR/examples")
[ -s "$RES" ] || printf "case\tresult\trc\tseconds\tfirst error\n" > "$RES"
echo "checkout: $ROLL_DIR ($(git -C "$ROLL_DIR" log -1 --format='%h %s' 2>/dev/null))"

grep -v '^#' "$HERE/cases.list" | while IFS='|' read -r name launcher cfg kind limit; do
  [ -z "$name" ] && continue
  [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$name" && continue
  [ -f "$WORK/STOP" ] && { echo "STOP file found"; break; }
  ray stop --force >/dev/null 2>&1
  rm -rf "$WORK/checkpoints" "$WORK/out" "$WORK/output" "$WORK/tensorboard" "$WORK/render"
  log=$LOGS/$name.log; t0=$(date +%s)
  ( cd "$ROLL_DIR" && timeout -k 60 "$limit" python examples/"$launcher" --config_path "$CREL" --config_name "$cfg" ) \
    > "$log" 2>&1 < /dev/null
  rc=$?; secs=$(( $(date +%s) - t0 ))
  err=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -m1 -E "^[A-Za-z_.]*(Error|Exception): " | cut -c1-160)
  if [ "$kind" = rlvr ]; then
    [ $rc = 0 ] && grep -q "pipeline complete" "$log" && st=PASS || st=FAIL
  else
    [ $rc = 0 ] && [ -z "$err" ] && st=PASS || st=FAIL
  fi
  { [ $rc = 124 ] || [ $rc = 137 ]; } && st="FAIL(time limit ${limit}s)"
  printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$st" "$rc" "$secs" "$err" | tee -a "$RES"
done
ray stop --force >/dev/null 2>&1
echo "done: $RES"
