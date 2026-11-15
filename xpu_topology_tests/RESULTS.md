# ROLL topology / recipe cases on 2 x Arc Pro B70 — 2026-11-14

Code: ROLL `af86fd4` (Madhustat/ROLL `xpu/explicit-device-placement-20261112`, cleaned explicit placement).
Runner: `run_suite.sh` + `topology.list`.
Per-case evidence: `work/runs/<case>/summary.txt`; table: `work/results.tsv`. 0 GPU faults, health OK after every case.

**Result: 61 PASS / 5 FAIL of 66** (SFT counted once, as the passing `SFT_1xpu_val500` variant).

| Group | Result |
|---|---|
| Separated (train+ref XPU0, vLLM XPU1), 33 incl. resume save/load | 32 PASS, LORA FAIL |
| vLLM TP=2 across both XPUs | PASS |
| Colocated / one device (A0, B1, W1b colocated, B1+Qwen3, PPO colocated, 1.5B offload B1) | 6/6 PASS |
| DPO (FSDP2 ref + reshard, 1 XPU, 2 XPU, retry) | 4/4 PASS |
| SFT 1 XPU | PASS with validation on 500 rows (`SFT_1xpu_val500`, 34 min); original validates on 20k rows (~22 h), stopped at the time limit without error |
| Agentic (16) | 13 PASS; GiGPO, Sokoban native, Sokoban agent_runner FAIL |
| Distillation (on-policy separated/colocated, off-policy nccl-only/ray) | 4/4 PASS |
| VL Qwen2.5-VL-3B | FAIL |
| L3, 40 steps | PASS 40/40 |

Failures (none are XPU or placement related; all also failed on 2026-11-04/05 before the fix):

| Case | Cause |
|---|---|
| LORA | `QKVParallelLinear has no attribute base_layer` at the first weight sync: ROLL keeps `.base_layer.` names for vLLM LoRA wrappers; ROLL has vLLM patches only for 0.11/0.12/0.17, the env runs 0.31 |
| AG_frozen_lake_gigpo, AG_sokoban_native | ROLL bug `roll/pipeline/agentic/utils.py:80`: `.astype` on a torch Tensor |
| AG_sokoban_agent_runner | `No module named 'sglang'` (ProxyEnvManager needs SGLang) |
| VL_3b_math | `Qwen2_5_VLModel.get_rope_index() missing 'mm_token_type_ids'`: ROLL's VL pipeline uses the older Transformers API, env runs Transformers 5.17 |
