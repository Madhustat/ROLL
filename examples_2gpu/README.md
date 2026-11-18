# ROLL examples on 2 GPUs

ROLL example configs reduced to two accelerators (FSDP2 training, Qwen2.5-0.5B / 1.5B-Instruct or Qwen3-0.6B,
4 to 16 training steps), plus a runner that executes them one at a time. The configs are device-neutral; the same
files run on NVIDIA and Intel XPU.

## Contents

- `configs/`: 38 configs
  - 17 workflow and layout cases: GRPO separated, colocated, and single-device; async generation; PPO with critic;
    vLLM tensor parallel (split and aligned layouts); DPO on 1 and 2 devices; agentic FrozenLake; TOPR with the critic
    on device 1; Qwen3-0.6B; Qwen2.5-1.5B with and without CPU offload; on-policy distillation
  - 21 upstream examples (`examples/docs_examples`, `qwen2.5-7B-rlvr_megatron`, `qwen2.5-7B-rlvr-offpolicy`,
    `qwen3-8B-rlvr_fsdp2`, `dpo_examples`, `qwen2.5-7B-sft_megatron`) converted to 2 devices, FSDP2, and small models
- `cases.list`: case name, launcher, config, kind, time limit
- `prepare.sh`: downloads the three models and CodeAlpaca-20k, and builds the data subsets from ROLL's `data/`
- `run_cases.sh`: runs the cases and writes a results table

## Run

```bash
export WORK=/path/to/work            # models, data, outputs, logs
./prepare.sh                         # once
./run_cases.sh                       # all cases, against the checkout this folder is in
./run_cases.sh base DPO_2gpu         # selected cases
ROLL_DIR=/path/to/other/ROLL ./run_cases.sh base   # same cases against another checkout, e.g. upstream main
touch $WORK/STOP                     # stop after the current case
```

A case passes when the launcher exits 0 and, for RLVR, the log contains `pipeline complete`; for the other kinds,
when it exits 0 with no Python exception in the log. Results go to `$WORK/results_<checkout>.tsv` and logs to
`$WORK/logs/<checkout>/`.

## Notes

- `TP2_split` puts training on device 0 and vLLM TP=2 on devices 0 and 1. On CUDA this layout selects the colocated
  CUDA-IPC update path, which requires training devices aligned with whole TP groups, so it fails the same way on
  upstream `main`. `TP2_aligned` (all roles on devices 0 and 1) is the CUDA tensor-parallel case.
- `q25_fsdp2_sp2_ga4` is the sequence-parallel example at 2048/4096 tokens and needs a large amount of host RAM;
  `q25_fsdp2_sp2_ga4_small` (512) and `q25_fsdp2_sp2_ga4_1k` (1024) are the reduced versions.
- RLVR runs can skip a step when every response is masked (`Current step will be skipped`); that is ROLL behavior.
