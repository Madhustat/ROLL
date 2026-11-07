#!/usr/bin/env bash
# CPU job: print the summary and the key error lines of a table run. usage: show_unit_logs.sh [run_dir] [log ids...]
D=${1:-$(ls -d /workspace/remote/unit_tests/*_table | tail -1)}; shift; IDS=${*:-03_try3 04_try1 05_try1 06_try1 08_try1 12_try1 15_try1 18_try1}
echo "run: $D"; cat $D/summary.tsv
for f in $IDS; do
  echo "===== $f: $(grep -m1 -oE 'tests/[^ ]+' $D/$f.log)"
  sed "s/\x1b\[[0-9;]*m//g" $D/$f.log | grep -E "SKIPPED \[|Error|Exception|raise |^E  |FAILED |Timeout|most recent call first|line [0-9]+ in " | grep -vE "Warning|warn" | awk '!s[$0]++' | head -40
done
echo "SHOW DONE"
