# ROLL FSDP log-prob unit tests with the XPU count reduced from 8 to 2: resource-preserving report

Date: 2026-11-10. Machine for all runs in this report: **local 2 × Intel Arc Pro B70 (32 GB)**. The same procedure is prepared for devctl 8 × B60 but has not been run there yet (see section 12).

## 1. Constraint

Only the XPU count was reduced. Changed fields in the two test YAMLs: `device_mapping` (two roles) and `fsdp_size`. Nothing else: no ROLL Python code, no Python test files, no feature settings, no local workaround variables (`ROLL_FSDP2_INIT_WITHOUT_BROADCAST` and `ROLL_LOGITS_CHUNK_SIZE` were explicitly set to `0` in every run; the ROLL tree is upstream `d3506ea9` + XPU enablement, in which both variables are inert at `0`).

## 2. Original 8-XPU configuration (both tests)

| Role | Devices | Sharding | Other |
|---|---|---|---|
| `actor_train` (policy, FSDP2) | GPUs 0-7 | `fsdp_size: 8` | `full`: `offload_policy: true`; `lora`: `offload_policy: False`, LoRA r=16 on q/k/v/o |
| `actor_infer` (vLLM, never started by the test) | GPUs 0-7 | - | - |
| `reference` (HF, single device) | GPU 0 | - | - |

Model Qwen3-0.6B (bf16, tied embeddings), prompt 128 / response 512, rollout batch 128, `infer_batch_size` 2.

## 3. Exact two-XPU diff (`devctl_roll/tests_logprobs_2xpu.patch`)

Identical for `log_probs_fsdp_config.yaml` and `log_probs_fsdp_lora_config.yaml`:

```
-      fsdp_size: 8
+      fsdp_size: 2
-  device_mapping: list(range(0,8))      # actor_train
+  device_mapping: list(range(0,2))
-  device_mapping: list(range(0,8))      # actor_infer (unused by the test)
+  device_mapping: list(range(0,2))
```

Unchanged: offload settings, model, LoRA settings, precision, batch and sequence sizes, data, comparison, FSDP behaviour apart from world size.

Launch settings (not test content): Ray head started with `--num-gpus 2`; `RAY_ADDRESS=127.0.0.1:6392` and `RAY_TMPDIR` so the test's `ray.init()` joins that head (the local venv has no `dpctl`, so a self-started Ray would detect 0 XPUs). oneCCL 2022.0.0 preloaded; `CCL_SYCL_*_SIMPLE_THRESHOLD` = 8 GiB (the standing proven settings). Placement check before each run: Ray reports exactly 2 GPUs, both B70 visible.

## 4. `test_fsdp_log_probs_lora`: clean pass, 3 of 3

| Attempt | Result | Duration | Peak reserved per XPU | Rows | FSDP2 vs reference: max per-token diff | mean diff |
|---|---|---|---|---|---|---|
| 1 | PASSED | 101 s | 2.26 GB | 128 | 0.612 | 0.034 (row max), 0.008 (overall mean) |
| 2 | PASSED | 82 s | 2.26 GB | 128 | 0.612 | same |
| 3 | PASSED | 82 s | 2.26 GB | 128 | 0.612 | same |

The test has no tolerance and no assertion; it records the differences (`test_fsdp_log_probs_lora.json`, copies in `clean2xpu/lora_{1,2,3}.json`). Metrics were identical across the three runs. Adapter enabled vs disabled difference = 0, expected for freshly initialised LoRA (B = 0). All 6 workers (2 policy, 1 reference, plus Ray bookkeeping) exited cleanly each time.

**Classification: clean pass** (device count reduced, functionality unchanged).

## 5. `test_fsdp_log_probs_full` with original CPU offload: hangs

| Attempt | Result | Where |
|---|---|---|
| 1 | HANG (killed at 900 s) | both policy ranks in `torch/distributed/_state_dict_utils.py:636` (`_distribute_tensors`, copy of each broadcast tensor into the local shard), called from `set_model_state_dict(..., broadcast_from_rank0=True)` at `roll/utils/fsdp_utils.py:417`; reference never started (waits for the policy) |
| 2 | HANG (killed at 900 s) | identical frames |

Kernel log during the hang: 0 GPU faults. Killing the hung ranks produced engine resets on both GPUs; health check passed afterwards each time. Evidence: `clean2xpu/full_{1,2}.log`, `full_{1,2}_pyspy_*.txt`, `full_{1,2}_raylogs/`.

**Classification: blocked, not a ROLL logic failure** (see section 7).

## 6. Diagnostic: `full` with `offload_policy: False`, non-final

One run, PASSED in 72 s, metrics identical to the LoRA runs (max diff 0.612). This changed a functional feature (CPU offload) and is diagnostic evidence only. The line was reverted afterwards.

## 7. ROLL-free reproducer (`clean2xpu/repro_offload_bcast.py`)

PyTorch only, 2 ranks, Qwen3-0.6B: all ranks load the model on CPU in bf16 (as ROLL does for tied-embedding models), `fully_shard` per decoder layer + root with `CPUOffloadPolicy` (or none), rank 0 `.to("xpu")`, rank 1 `.to_empty(device="xpu")`, then `set_model_state_dict(model, full_state, StateDictOptions(full_state_dict=True, cpu_offload=<on/off>, broadcast_from_rank0=True))`. 5-minute hard timeout, stack dump at 200 s.

| Run | CPU offload | Result | Last stage reached |
|---|---|---|---|
| off_1, off_2, off_3 | off | **PASS**, 2.1 s / 2.0 s / 2.6 s | `state_dict_set`, `LOAD_OK` |
| on_1, on_2, on_3 | on | **HANG**, killed at 300 s each | `sharded` (2.1 s); all stuck in `_state_dict_utils.py:636 ← 595 ← 681 ← state_dict.py:609`, the same frames as ROLL |

Deterministic: 3 of 3 hang with CPU offload, 3 of 3 pass without, nothing else changed between the two cases. Logs: `clean2xpu/repro_offload_{on,off}_*_r{0,1}.log`. GPUs recovered after every kill (health check passed).

### Full two-rank matrix (completed)

| CPU offload | `broadcast_from_rank0` | Runs | Result |
|---|---|---|---|
| on | True (as ROLL) | 3 | **hang 3 of 3** |
| off | True | 3 | pass 3 of 3 (2.0-2.6 s) |
| on | False (every rank loads its checkpoint, local distribute) | 3 | pass 3 of 3 (3.0-3.4 s) |
| off | False | 1 | pass (2.4 s) |

Synthetic diagnostic (`clean2xpu/diag_copy_bcast.py`, 2 ranks, synchronize + barrier after every step, sizes 296 MB / 6 MB / 2 MB): pageable H2D, pinned H2D, XCCL broadcast, broadcast followed by D2H slice copies into pageable and pinned memory, second-half slice: **all steps pass on both ranks**.

Conclusion: the hang requires CPU offload **and** the rank-0 broadcast together, inside PyTorch's `set_model_state_dict` → `_broadcast_state_dict` → `_distribute_tensors` path. Neither alone hangs, and a plain broadcast followed by a device-to-host copy does not hang.

### 7a. Environment-setting scan on the failing reproducer (CPU offload on, 2 ranks, one variable per run)

| Setting | Result | Note |
|---|---|---|
| none (baseline) | hang 3 of 3 | |
| `TORCH_XCCL_BLOCKING_WAIT=1` | hang | 90 s limit; stuck 2 s after start |
| `CCL_SYCL_OUTPUT_EVENT=1` | hang | 90 s limit; stuck 2 s after start |
| `CCL_ZE_ENABLE=0` | not applicable | XCCL fails to create a communicator without Level Zero (`onecclCommInitRank failed with code 11`) |
| `CCL_BCAST=ring` | hang | 300 s limit |
| `CCL_ZE_IPC_EXCHANGE=sockets` | hang | 300 s limit |
| `CCL_ZE_COPY_ENGINE=none` | hang | 300 s limit |
| `CCL_ZE_DEPS_SYNC=1` | hang | 300 s limit |
| `CCL_ATL_TRANSPORT=ofi` | no change | already the default: oneCCL log on a passing run shows `provider: tcp`, `libfabric 2.4.0-impi` |
| `FI_PROVIDER=tcp` | no change | already the default provider (see above); `fi_info` not usable here |

Every hung run was killed by the timeout; the kernel log shows engine resets within 3 s of each kill (page fault response `-EINVAL`, `bcs`/`ccs` resets, one `CAT` error) and, after the second kill, a GPU 1 scheduler timeout with a device coredump and a full GT reset at 05:17:54 that recovered on its own. The health check passed after every cycle once the reset completed. The faults follow the kills, not the settings. Two more delayed GT resets followed later kills (GPU 0 at 06:02:18, GPU 1 at 06:04:18, 4-6 min after the kills at 05:57:53); a run started inside that window hung as collateral and was discarded, and the plain offload-off baseline passed again at 06:06 (2.0 s), confirming the GPUs were otherwise fine. Every genuine setting hang began about 2 s into the run, before any such reset.

**Result of the scan: no environment setting avoids the hang.**

### Property comparison, ROLL `full` test vs reproducer

| Property | ROLL `full` | Reproducer | Match |
|---|---|---|---|
| Model / dtype | Qwen3-0.6B, bf16 | same | yes |
| World size | 2 (2-XPU variant) | 2 | yes |
| Tied embeddings → all ranks load real weights on CPU (`use_meta_tensor = not tie_word_embeddings`) | yes | yes | yes |
| FSDP2 wrapping | per `Qwen3DecoderLayer` + root (`apply_fsdp2`) | per `model.model.layers[i]` + root | yes |
| `CPUOffloadPolicy` | from `offload_policy: true` | `CPUOffloadPolicy()` | yes |
| Rank-0 `.to(device)`, others `.to_empty(device)` | yes (rank 0 with `non_blocking=True`) | yes (blocking) | yes, apart from `non_blocking` |
| `StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True)` | yes | yes | yes |
| Broadcast bucket | PyTorch's 500 bytes → about one broadcast per tensor (311 tensors, 1.4 GiB) | same call | yes |
| Process group | `cpu:gloo,xpu:xccl`, no `device_id` | `cpu:gloo,xpu:xccl` with `device_id=xpu:0` | not exact |
| Device selection | `ZE_AFFINITY_MASK=<rank>` per Ray worker | `ZE_AFFINITY_MASK=<rank>` per process | yes |
| `MixedPrecisionPolicy` (param bf16, reduce fp32) | yes | not set (model already bf16) | not exact |
| Ray | workers are Ray actors | plain processes | no (by design: shows Ray is not needed) |

The important properties (model, offload policy, state-dict options, broadcast path, backend, device mapping) match; the hang and the stuck frames are the same.

## 8. Versions (local machine)

| Component | Version |
|---|---|
| PyTorch | 2.14.0+xpu, git `08187d9e0fba` |
| oneCCL runtime | **2022.0.0** preloaded (`LD_PRELOAD` of `libccl.so.2`); pip package in the venv is `oneccl 2022.1.2` and is overridden at load. XCCL is built into PyTorch; there is no `oneccl-bind-pt` |
| SYCL runtime | `intel-sycl-rt 2026.1.1`, `dpcpp-cpp-rt 2026.1.1`, `umf 1.1.0` |
| NEO (compute runtime) | 26.31.39395.14 (`libze-intel-gpu1`, `intel-opencl-icd`), driver reported `1.17.39395+14` |
| Level Zero loader | system `libze1` 1.32 |
| Ray / Transformers / Python | 2.48.0 / 5.17.0 / 3.12.3 |
| OS / kernel | Ubuntu 24.04.2, 7.0.0-42-generic |
| vLLM (installed, not used by these tests) | 0.31.0+xpu |

devctl differences (from earlier runs): NEO 26.14.37833.4, Level Zero loader 1.28.2, kernel 7.0.0-38, image `intel/pytorch:xpu-2.11.0-ubuntu24.04` (digest `sha256:dda613c2…`, as pulled 2026-11-07). The "mixture" is by design: the venv is built by `setup_env.sh` inside the container with PyTorch 2.14.0+xpu and vLLM 0.31; the image's own PyTorch 2.11 is not used and not modified. Nothing was upgraded or downgraded during diagnosis.

## 9. `cp` and `cp_rmpad`: resource-blocked

From code: `test_fsdp_log_probs_cp` skips below 8 devices (`test_fsdp_log_probs.py:377`); both configs use `fsdp_size: 4` × `ulysses_size: 2` = 8, `prompt_length: 8192` padded to full length, and a single-device HF reference. Measured reference-path peak without chunking: 17.3 GiB at 2 × 4,096 tokens, out of memory at 2 × 8,192 on a 32 GB B70 (about 34 GB needed); B60 has 24 GB. Preserving the original topology needs 8 devices with more than about 34 GB each. Not executed; not counted as passes; not ROLL failures.

## 10. Would three XPUs help?

No. From code and evidence: the `full` blocker is the CPU-offload state-dict broadcast, which hangs already at 2 ranks and reproduces without ROLL; it is not a world-size problem. `fsdp_size: 3` is technically allowed (FSDP2 pads uneven shards; ROLL's `DataProto.chunk` splits unevenly), but it changes nothing about the blocker and adds exposure to the rank-count-dependent oneCCL broadcast hang seen on devctl (0 of 10 at 2 ranks, 2 of 10 at 4, 4-6 of 10 at 8). Recommendation: stay with 2 XPUs.

## 11. Ownership

| Item | Classification | Owner |
|---|---|---|
| `lora` 2-XPU | clean pass | - |
| `full` 2-XPU | platform-stack blocked: reproduces without ROLL; needs CPU offload **and** rank-0 broadcast together in PyTorch's `set_model_state_dict`; no environment setting avoids it | PyTorch-XPU / oneCCL stack; bug report to be filed with `repro_offload_bcast.py` |
| `full` 8-XPU, `lora` 8-XPU on devctl | platform-stack blocked (rank-count-dependent oneCCL broadcast hang, ROLL-free reproducer `repro_bcast.py`) | oneCCL |
| `cp`, `cp_rmpad` | resource-blocked (8 devices > 34 GB each) | hardware allocation |
| `full` with offload off | diagnostic pass only | - |

## 12. Answers

1. **Does LoRA pass cleanly on two XPUs with functionality unchanged?** Yes: 3 of 3, identical metrics, only `device_mapping` and `fsdp_size` changed.
2. **Does `full` pass cleanly on two XPUs with CPU offload unchanged?** No: it hangs at the start-up weight broadcast, 2 of 2.
3. **Does the `full` hang reproduce without ROLL?** Yes: the PyTorch-only reproducer hangs 3 of 3 with CPU offload and passes 3 of 3 without, at the same frames as ROLL (section 7).
4. **Would three XPUs solve it?** No; the blocker is independent of world size and more ranks increase exposure to the oneCCL broadcast hang.
5. **Can `cp` / `cp_rmpad` keep their functionality on fewer than 8 devices?** No: 8 devices are required by code and topology, and the unchanged reference path needs more than 34 GB per device.
6. **Which failures belong to ROLL?** None of these. `full` and the 8-XPU start-ups are platform-stack blockers; `cp` / `cp_rmpad` are resource blockers.

Pending (need a decision): (a) the reproducer on the stock `intel/pytorch:xpu-2.11.0` container stack (PyTorch 2.11 + oneCCL 2021.17.2, as shipped together); (b) the reproducer and the 2-XPU `full` test on 2 CRI GPUs; (c) filing the PyTorch-XPU/oneCCL issue with `repro_offload_bcast.py`; (d) repeating the 2-XPU procedure on devctl 8 × B60.
