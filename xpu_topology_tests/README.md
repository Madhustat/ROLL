# ROLL topology and recipe tests on 2 Intel XPUs

68 ROLL pipeline cases that cover the device layouts and training recipes on 2 XPUs, run one at a time by
`run_suite.sh`. The base case is Qwen2.5-0.5B GRPO with train + reference on XPU 0 and vLLM on XPU 1; every other case
changes one part of it.

| Group | Cases | What changes |
|---|---|---|
| Separated layout | 34 | weight sync on/off, async, loss and advantage variants, Reinforce++, PG losses (TOPR, TIS, CISPO, Kimi 1.5), PPO / LitePPO with critic, query filter, Qwen3-0.6B, Qwen2.5-1.5B (with and without CPU offload), LLM judge, TransferQueue, profiler, checkpoint save then resume, LoRA, vLLM TP=2, TOPR with the critic on XPU 1 |
| Colocated / one device | 6 | all roles on both XPUs or on one XPU (CPU-staging weight sync), including PPO and 1.5B offload |
| DPO / SFT | 5 | 1 and 2 XPUs, FSDP2 reference with reshard |
| Agentic | 17 | FrozenLake and Sokoban: disaggregated, colocated, partial GPU, async, GiGPO, TOPR, PPO, SAO, partial rollout, rollout only, dynamic batching |
| Distillation | 4 | on-policy (separated, colocated), off-policy (`nccl-only` and `ray` logits transfer) |
| Vision-language | 1 | Qwen2.5-VL-3B math |
| Long run | 1 | 40 steps |

## Get and run

```bash
git clone -b validation/final-xpu-suites-20261124 https://github.com/Madhustat/ROLL.git
cd ROLL/xpu_topology_tests
VENV=... ONECCL_LIB=... MODELS_DIR=... DATA_DIR=... EXTRA_DATA_DIR=... ./run_suite.sh
```

```bash
./run_suite.sh                         # every case in topology.list, in the background
./run_suite.sh base A0_colocated       # selected cases
RESUME=1 ./run_suite.sh                # skip cases whose latest result is PASS
tail -f work/suite.out                 # follow progress
touch work/STOP                        # stop after the current case
```

## Settings (environment variables)

Defaults are the paths on the original test machine.

| Variable | Meaning |
|---|---|
| `ROLL_DIR` | ROLL checkout (default: the checkout this folder is in) |
| `VENV` | Python environment |
| `ONECCL_LIB` | oneCCL 2022.0.0 `lib` directory (preloaded) |
| `MODELS_DIR` | models: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen3-0.6B, Qwen2.5-VL-3B-Instruct |
| `DATA_DIR` | `gsm8k_zh/GSM8K_zh.json` (off-policy distillation) and `orsta/train/train_math_mmmath_3539.parquet` (VL); the RLVR cases read `data/math_deepmath_deal.jsonl` from the ROLL checkout |
| `EXTRA_DATA_DIR` | DPO / SFT data: `comparison_gpt4_data_zh.json`, `comparison_gpt4_val64.json`, `code_alpaca_500.json`, `code_alpaca_20k.json` |
| `WORK` | output directory (default `work/`) |
| `MIN_RAM_GB` | stop a case when available host RAM drops below this many GB (default 10) |

The configs read `${oc.env:ROLL_WORK}` (set to `work/rw`, which links `models`, `data` and `extra_data` to the
directories above and holds outputs and checkpoints) and `${oc.env:ROLL_REPO}` (the ROLL checkout).

## What the runner does

- **Before each case:** wait until no process uses the GPUs; clean outputs, checkpoints, compile caches and temp; GPU
  health check.
- **During:** watchdog for the time limit, a stalled log, xe kernel GPU faults and low host RAM.
- **After:** stop the case's processes and Ray; a 10-minute cool-down if a GPU fault appeared; health check.
- **Resume pair:** `RESUME_a_save` keeps its checkpoint, and `RESUME_b_load` loads it, then the checkpoint is removed.
- **Results:** `work/results.tsv`, plus `work/runs/<case>/summary.txt` for each case (a few KB). The full log is kept
  as `run.log.gz` only when a case does not pass.

Latest results: [RESULTS_20261123.md](RESULTS_20261123.md) (combined branch, 2026-11-23). Earlier: [RESULTS.md](RESULTS.md).
