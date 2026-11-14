# ROLL upstream examples on 2 Intel XPUs

The upstream ROLL example pipelines, converted to 2 XPUs, run one at a time by `run_suite.sh`. The configs in
`configs/` use the placeholders `__MODELS_DIR__`, `__DATA_DIR__` and `__OUT_DIR__`; the runner fills them in.

## Get and run

```bash
git clone -b xpu/explicit-device-placement-20261112 https://github.com/Madhustat/ROLL.git
cd ROLL/xpu_upstream_examples
VENV=... ONECCL_LIB=... MODELS_DIR=... DATA_DIR=... ./run_suite.sh
```

```bash
./run_suite.sh                    # every example in examples.list, in the background
./run_suite.sh ex_grpo ex_ppo     # selected examples
RESUME=1 ./run_suite.sh           # skip examples whose latest result is PASS
SUITE_FG=1 ./run_suite.sh ex_ppo  # stay in the foreground
tail -f work/suite.out            # follow progress
touch work/STOP                   # stop after the current example
```

## Settings (environment variables)

Defaults are the paths on the original test machine.

| Variable | Meaning |
|---|---|
| `ROLL_DIR` | ROLL checkout (default: the checkout this folder is in) |
| `VENV` | Python environment |
| `ONECCL_LIB` | oneCCL 2022.0.0 `lib` directory (preloaded) |
| `MODELS_DIR`, `DATA_DIR` | models and datasets |
| `WORK` | output directory (default `work/`) |
| `KERNELS_SEED` | Hugging Face kernels cache copied once into `WORK` |
| `MIN_RAM_GB` | stop an example when available host RAM drops below this many GB (default 10) |
| `SUITE_TMP` | temp directory (default `/tmp/roll_suite_tmp_$USER`; keep it short, Unix socket paths are limited) |

## What the runner does

- **Before each example:** wait until no process uses the GPUs, clean caches / checkpoints / outputs / temp, GPU health
  check.
- **During:** watchdog for the time limit, a stalled log, xe kernel GPU faults and low host RAM.
- **After:** stop the example's processes and Ray; a 10-minute cool-down if a GPU fault appeared; health check.
- **Robustness:**
  - The runner detaches from the terminal, so closing the shell does not stop it.
  - Only one suite runs at a time.
  - At start it removes leftovers of an earlier killed run, but only processes it tagged itself.
  - If the runner is killed, it stops the current example and Ray first.

## Results

- `work/results.tsv` has one line per example: result, rc, steps, seconds, GPU faults, health, worker placement and
  the first error.
- `work/runs/<example>/summary.txt` is a few KB: the commit, placement, steps, peak GPU memory, errors and the last
  lines. The full log is deleted for a PASS and kept as `run.log.gz` otherwise.
- **RLVR pass rule:** ROLL logs `pipeline complete` and exits 0, and every step either logs `pipeline step N finished`
  or was skipped by ROLL itself. ROLL skips a step (`Current step will be skipped`) when `max_len_mask` /
  `difficulty_mask` mask every response, which happens at random with the small 2-XPU batches; it is reported as
  `PASS(<n> step skipped by ROLL ...)`.
- The per-step metric JSON lines are shown in the summary but are not used for the pass rule.

## Configs changed beyond the 2-XPU conversion

| Config | Change | Why |
|---|---|---|
| `ex_topr_small` | prompt/response 512/512, `max_model_len` 1024 (original 6144 tokens) | at 6144 tokens actor and critic training together exceed one 32 GB XPU |
| `q25_fsdp2_sp2_ga4` | `gradient_accumulation_steps: 4` | the original fails ROLL's batch-size check on 2 XPUs |
| `q25_fsdp2_sp2_ga4_1k` | as above, prompt/response 1024/1024, `max_model_len` 2048 | 2048/4096 runs a 125 GB host out of RAM |
| `q25_fsdp2_sp2_ga4_small` | as above, prompt/response 512/512, `max_model_len` 1024 | same |

Results on 2 × Arc Pro B70 (2026-11-13): every example PASS, including both reduced `q25_fsdp2_sp2_ga4` variants. The
only failure is `q25_fsdp2_sp2_ga4` at 2048/4096: the host ran out of RAM.
