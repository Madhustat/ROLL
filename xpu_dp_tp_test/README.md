# vLLM data parallel with tensor parallel on Intel XPUs

One ROLL GRPO run with vLLM `data_parallel_size: 2` and `tensor_parallel_size: 2` together, to check that every vLLM
worker runs on the XPU ROLL assigned to it. It needs at least 4 XPUs, so it does not fit the 2-XPU suites in
`xpu_topology_tests/` and `xpu_upstream_examples/`.

| Config | XPUs needed | Layout | Expected vLLM engines |
|---|---|---|---|
| `moe_dp2_tp2_4xpu` | 4 | training and reference share XPU 0 with vLLM | engine 0 on XPUs 0, 1; engine 1 on XPUs 2, 3 |
| `moe_dp2_tp2` | 5 | training and reference on XPU 0 | engine 0 on XPUs 1, 2; engine 1 on XPUs 3, 4 |
| `moe_dp2_tp2_rev` | 5 | as above, vLLM device mapping `[4, 3, 2, 1]` | engine 0 on XPUs 4, 3; engine 1 on XPUs 2, 1 |

Model: the public tiny random MoE model `yujiepan/qwen3-moe-tiny-random` (vLLM 0.31 accepts external data parallel
only for MoE models). The weights are random, so only placement and completion are meaningful.

## Run

`run_dp_tp.sh` is written for a job container where this checkout is at `/workspace/local/ROLL_dp_tp`, this folder is
copied to `/workspace/local/devctl_roll/dp_tp`, and `/workspace/local/devctl_roll/scripts/env_roll_devctl.sh` sets up
the Python environment (PyTorch XPU, vLLM 0.31.0+xpu, Ray, oneCCL 2022.0.0 preloaded). Change the three paths at the
top of the script for another machine.

```bash
# once: download the model to $ROLL_WORK/models/qwen3-moe-tiny-random
python -c "from huggingface_hub import snapshot_download as s; s('yujiepan/qwen3-moe-tiny-random', local_dir='$ROLL_WORK/models/qwen3-moe-tiny-random')"

# one run (about 20 minutes; stops itself after 25 minutes)
bash run_dp_tp.sh moe_dp2_tp2_4xpu
```

The script prints a result block: PASS / FAIL / HANG, steps finished, ROLL's `XPU placement:` lines, peak GPU memory per
process and per GPU (from `/proc/<pid>/fdinfo`), and counts of out-of-bounds errors, narrow-mask warnings and
tracebacks. A pass needs each `VLLM::Worker_DPx_TPy` process to hold its memory on its assigned XPU.

Measured result: [RESULT_20261124.md](RESULT_20261124.md).
