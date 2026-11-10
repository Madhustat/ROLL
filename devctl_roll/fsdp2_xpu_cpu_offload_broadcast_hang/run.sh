#!/usr/bin/env bash
# usage: run.sh [on|off] [on|off] [model]   -> CPU offload, rank-0 broadcast, model path. Two ranks on XPU 0 and 1, 300 s limit.
OFF=${1:-on}; BC=${2:-on}; M=${3:-Qwen/Qwen3-0.6B}; P=$((29600 + RANDOM % 300)); D=$(dirname "$0")
for r in 0 1; do ( ZE_AFFINITY_MASK=$r timeout 300 python "$D/repro.py" --rank $r --world 2 --port $P --offload $OFF --broadcast $BC --model "$M"; echo "EXIT rc=$?" ) > "$D/rank$r.log" 2>&1 & done; wait
grep -q LOAD_OK "$D/rank0.log" && grep -q LOAD_OK "$D/rank1.log" && echo "PASS offload=$OFF broadcast=$BC" || { echo "HANG/FAIL offload=$OFF broadcast=$BC (rc 124 = killed at 300 s)"; grep -oE "STAGE [a-z_]+" "$D/rank0.log" | tail -1; }
