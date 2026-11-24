"""Per-step RLVR metrics from a ROLL run.log (the JSON metric dicts the stdout tracker prints). Usage: rlvr_metrics.py run.log"""
import json
import re
import sys

KEYS = [("step", "system/global_step"), ("total_s", "time/step_total"), ("gen_s", "time/step_generate"),
        ("train_s", "time/step_train"), ("upd_s", "time/step_model_update"), ("score", "critic/score/mean"),
        ("kl_loss", "actor/kl_loss@sum"), ("train_infer_kl", "actor/train_infer_kl@sum"),
        ("entropy", "critic/entropy/mean"), ("resp_len", "token/response_length/mean"),
        ("clipped", "token/total_response_length/clip")]

text = re.sub(r"\x1b\[[0-9;]*m", "", open(sys.argv[1], errors="replace").read())
decoder = json.JSONDecoder()
seen = {}
for m in re.finditer(r"\{\"", text):
    try:
        d, _ = decoder.raw_decode(text, m.start())
    except ValueError:
        continue
    if isinstance(d, dict) and "system/global_step" in d:
        seen[int(d["system/global_step"])] = d
checkpoints = {int(n) for n in re.findall(r"/checkpoint-(\d+) save to ", text)}
finished = {int(n) for n in re.findall(r"pipeline step (\d+) finished", text)}
started = {int(n) for n in re.findall(r"pipeline step (\d+) start\.\.\.", text)}
skipped = len(re.findall(r"Current step will be skipped", text))
print(" ".join(f"{k:>14}" for k, _ in KEYS))
for step in sorted(seen):
    d = seen[step]
    print(" ".join(f"{d.get(v, float('nan')):>14.6g}" if isinstance(d.get(v), (int, float)) else f"{'-':>14}" for _, v in KEYS))
print(f"steps logged: {len(seen)}")
print(f"last checkpoint step: {max(checkpoints) if checkpoints else -1}")
print(f"steps finished: {len(finished)}")
print(f"steps skipped by ROLL (all responses masked): {skipped}")
print(f"steps completed: {min(len(started), len(finished) + skipped)}")
