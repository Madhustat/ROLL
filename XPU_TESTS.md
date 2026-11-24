# Intel XPU test folders on this branch

This branch is upstream `main` plus the device-neutral platform calls, the Intel XPU platform, explicit XPU device
placement and the async partial-rollout fix, with three test folders added. The test folders are not part of any
pull request.

| Folder | What it runs | Needs |
|---|---|---|
| [xpu_upstream_examples/](xpu_upstream_examples/README.md) | 21 upstream ROLL examples reduced to 2 XPUs | 2 XPUs |
| [xpu_topology_tests/](xpu_topology_tests/README.md) | 68 layout and recipe cases, including cases converted from upstream agentic, distillation and DPO configs | 2 XPUs |
| [xpu_dp_tp_test/](xpu_dp_tp_test/README.md) | vLLM data parallel 2 with tensor parallel 2 | 4 or 5 XPUs |
| [examples_2gpu/](examples_2gpu/README.md) | the 2-GPU cases used for the NVIDIA no-regression check | 2 GPUs |

## Run on 2 XPUs

```bash
git clone -b validation/final-xpu-suites-20261124 https://github.com/Madhustat/ROLL.git && cd ROLL
export VENV=...          # Python environment: PyTorch XPU, vLLM 0.31.0+xpu, Ray
export ONECCL_LIB=...    # oneCCL 2022.0.0 lib directory (preloaded by the runner)
export MODELS_DIR=... DATA_DIR=... EXTRA_DATA_DIR=...   # see each folder's README for the contents

(cd xpu_upstream_examples && SUITE_FG=1 ./run_suite.sh)          # all 21 examples, about 5 hours
(cd xpu_topology_tests && SUITE_FG=1 ./run_suite.sh)             # all 68 cases
(cd xpu_topology_tests && SUITE_FG=1 ./run_suite.sh base A0_colocated TP2_vllm_2xpu)   # selected cases
(cd xpu_topology_tests && RESUME=1 SUITE_FG=1 ./run_suite.sh)    # only cases whose latest result is not PASS
```

Notes from the 2026-11-23 run:

- Run `RESUME_a_save` and `RESUME_b_load` in the same `run_suite.sh` call; the saved checkpoint is removed when the
  call ends.
- The runner stops a case when available host RAM drops below `MIN_RAM_GB` (default 10). On a 125 GB machine the
  colocated, TOPR and `q25_fsdp2_sp2_ga4*` cases passed when about 110 GB was free at the start and were stopped when
  about 80 GB was free.
- Only one suite runs at a time; a second call exits with "another suite is running".

Results of the 2026-11-23 run on this code: [xpu_topology_tests/RESULTS_20261123.md](xpu_topology_tests/RESULTS_20261123.md).
