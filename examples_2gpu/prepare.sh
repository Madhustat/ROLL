#!/usr/bin/env bash
# Download the models and build the data files the 2-GPU cases use. usage: WORK=<dir> ./prepare.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROLL_DIR=${ROLL_DIR:-$(cd "$HERE/.." && pwd)}
WORK=${WORK:-$HERE/work}
HF=$(command -v huggingface-cli || command -v hf || true)
[ -n "$HF" ] || { echo "huggingface-cli (or hf) not found: pip install -U huggingface_hub"; exit 1; }
mkdir -p "$WORK/models" "$WORK/extra_data" "$WORK/data/code_alpaca"

for m in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen3-0.6B; do
  "$HF" download "$m" --local-dir "$WORK/models/${m#*/}"
done
"$HF" download sahil2801/CodeAlpaca-20k code_alpaca_20k.json --repo-type dataset --local-dir "$WORK/data/code_alpaca"

python3 - "$ROLL_DIR" "$WORK" <<'EOF'
import json, shutil, sys
roll, work = sys.argv[1], sys.argv[2]
src = f"{roll}/data/comparison_gpt4_data_zh.json"
rows = json.load(open(src))
shutil.copy(src, f"{work}/extra_data/comparison_gpt4_data_zh.json")
json.dump(rows[:64], open(f"{work}/extra_data/comparison_gpt4_val64.json", "w"), ensure_ascii=False)
json.dump(rows[:2000], open(f"{work}/data/comparison_gpt4_data_zh_2000.json", "w"), ensure_ascii=False)
code = json.load(open(f"{work}/data/code_alpaca/code_alpaca_20k.json"))
json.dump(code[:80], open(f"{work}/data/code_alpaca/code_alpaca_80.json", "w"), ensure_ascii=False)
print("data ready:", len(rows), "comparison rows,", len(code), "code_alpaca rows")
EOF
echo "prepared $WORK"
