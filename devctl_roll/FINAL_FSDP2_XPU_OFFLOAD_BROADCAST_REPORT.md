# FINAL: FSDP2 CPU offload + rank-0 broadcast hang on Intel XPU (ROLL `test_fsdp_log_probs_full`)

Date: 2026-11-10. Measurements on local 2 × Intel Arc Pro B70 (32 GB) unless stated; the same hang was seen on devctl 8 × Arc Pro B60 through ROLL. Special experiment under the rule "only the XPU count may change"; every environment change was per-process and nothing persistent was altered.

## 1. Exact minimal failing condition

```
FSDP2 fully_shard(..., offload_policy=CPUOffloadPolicy())
+ set_model_state_dict(model, full_state, StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True))
+ process group "cpu:gloo,xpu:xccl", 2 ranks, Intel XPU
```

Reproduces with PyTorch only (no ROLL, no Ray): `fsdp2_xpu_cpu_offload_broadcast_hang/repro.py`. Both ranks stop in `torch/distributed/_state_dict_utils.py:636` (`_distribute_tensors`, copying each broadcast tensor into the CPU-offloaded local shard). Not inside `torch.xpu.synchronize()`. No GPU fault during the hang. Live stacks on Qwen2.5-1.5B show the ranks on **different tensors** (rank 1 still on `layers.0.self_attn.k_proj.weight`, rank 0 already on `v_proj`): the ranks fall out of step after the first few per-tensor broadcasts.

## 2. Passing controls (same model, same collective)

| CPU offload | rank-0 broadcast | Runs | Result |
|---|---|---|---|
| on | on | 3 | **hang 3 of 3** |
| off | on | 3 | pass (2.0–2.6 s) |
| on | off (every rank loads its checkpoint) | 3 | pass (3.0–3.4 s) |
| off | off | 1 | pass |

Synthetic checks that pass: pageable and pinned host-to-device copies, a plain XCCL broadcast, broadcast followed by a device-to-host slice copy (296 / 6 / 2 MB); one broadcast per tensor (500-byte buckets) without CPU offload.

## 3. Model matrix (offload on, broadcast on, 2 ranks)

| Model | Params | Tied | Tensors | Largest tensor | Total bf16 | Result |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 752 M | yes | 311 | `lm_head.weight` 297 MiB | 1.40 GiB | hang 3 of 3 |
| Qwen2.5-0.5B-Instruct | 494 M | yes | 290 | `embed_tokens` 260 MiB | 0.92 GiB | hang 1 of 1 |
| Qwen2.5-1.5B-Instruct | 1544 M | yes | 338 | `embed_tokens` 445 MiB | 2.88 GiB | hang 1 of 1 (300 s; live stacks captured) |

With PyTorch's 500-byte coalescing limit no two tensors share a bucket for any of these models (bucket count = tensor count), so the broadcast pattern is identical across models: one `dist.broadcast` per tensor. The hang is **not** model-size dependent. Details: `fsdp2_xpu_cpu_offload_broadcast_hang/model_matrix.md`.

## 4. Allocator variables

Neither `PYTORCH_ALLOC_CONF` nor `PYTORCH_CUDA_ALLOC_CONF` is set in our environment (`stage1_env.sh` and the devctl env script unset them). The baseline 3 of 3 hang therefore already covers "no expandable segments".

## 5. `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0` (alone)

**Hang** (1 run, killed at 300 s, same step; no GPU fault during the run). Run only on the reproducer, from a clean process environment with that single variable added; never written to an environment file (it belongs to a oneCCL knob block that produced GPU page faults on our systems in September 2026). Per the plan, environment-variable experimentation stops here.

Other settings tried one at a time, all hang: `TORCH_XCCL_BLOCKING_WAIT=1`, `CCL_SYCL_OUTPUT_EVENT=1`, `CCL_BCAST=ring`, `CCL_ZE_IPC_EXCHANGE=sockets`, `CCL_ZE_COPY_ENGINE=none`, `CCL_ZE_DEPS_SYNC=1`. `CCL_ZE_ENABLE=0` is unusable (XCCL cannot create a communicator). `CCL_ATL_TRANSPORT=ofi` and `FI_PROVIDER=tcp` are already the defaults (oneCCL log: `provider: tcp`, `libfabric 2.4.0-impi`).

Minimal pattern test (`clean2xpu/diag_bcast_d2h_loop.py`: 400 × {XCCL broadcast of one Qwen2.5-1.5B-layer-sized bf16 tensor; device-to-host copy of this rank's slice}, 2 ranks): **passes** both with the per-step copy (4 s) and without it (3 s). The plain broadcast-then-host-copy pattern alone does not hang; the hang needs PyTorch's FSDP2 state-dict path (`set_model_state_dict` → `_broadcast_state_dict` → `_distribute_tensors` on CPU-offloaded DTensor shards).

**Classification:** Platform-stack blocked: FSDP2 `CPUOffloadPolicy` combined with `set_model_state_dict(broadcast_from_rank0=True)` hangs on Intel XPU/XCCL. The same model and collective pass when either CPU offload or rank-0 broadcast is removed.

## 6. Comparison with the 4 November "Qwen2.5-1.5B + FSDP2 offload PASS"

| Property | 4 Nov pipeline (`nv_configs/v_1P5B_OFFLOAD.yaml`) | Failing `full` test / reproducer |
|---|---|---|
| FSDP version | FSDP2 (`fsdp2_train`) | FSDP2 |
| CPUOffloadPolicy on parameters | yes | yes |
| Checkpoint loading | all ranks on CPU (tied embeddings) | same |
| `broadcast_from_rank0` | True (upstream path) | True |
| Backend | `cpu:gloo,xpu:xccl` | same |
| Phase | training initialisation | same |
| State-dict API | `set_model_state_dict` | same |
| Init device | CPU | CPU |
| Tied weights | yes | yes |
| Stack | same venv: PyTorch 2.14.0+xpu, oneCCL 2022.0.0 preloaded | same |
| **World size of the FSDP2 group** | **1** (`actor_train` on `list(range(0,1))`) | **2** (and 8 on devctl) |

With one rank there is no inter-rank broadcast, so the failing condition was never exercised. The 4 November result is not contradictory, and the feature table's "FSDP2 CPU offload: Enabled" must be read as "1 GPU only". The only multi-rank offload example (`q25_fsdp2_sp2`, offload on the 2-rank reference role) never passed (1 error, 4 start-up hangs).

## 7. Stack versions (local)

PyTorch 2.14.0+xpu (git `08187d9e0fba`); XCCL built in (no `oneccl-bind-pt`); oneCCL runtime 2022.0.0 preloaded (pip `oneccl` 2022.1.2 overridden); `intel-sycl-rt` 2026.1.1; NEO 26.31.39395.14 (driver `1.17.39395+14`); Level Zero loader 1.32; Ubuntu 24.04.2, kernel 7.0.0-42. devctl: NEO 26.14, loader 1.28.2, kernel 7.0.0-38, same venv recipe. The stock `intel/pytorch:xpu-2.11.0` image (PyTorch 2.11.0+xpu, oneCCL 2021.17.2) was tried as a matched stack but could not complete even a plain two-process XCCL broadcast in our Docker set-up: inconclusive.

## 8. Related public issues (related evidence, not claimed to be the same root cause)

- intel/torch-xpu-ops **#5999**: FSDP multi-process tests hang until the 300 s harness timeout (duplicate of #5968).
- intel/torch-xpu-ops **#5968**: FSDP checkpoint / grad-scaler / unshard cases time out; label `dependency component: Level_Zero`; traced to a UR Level Zero adapter abort fixed upstream, awaiting a oneAPI pin bump.
- intel/torch-xpu-ops **#6092**: FSDP unshard and backward-reshard multi-process tests hang (regression).
- pytorch/pytorch **#202028**, **#196562**: FSDP2 `CPUOffloadPolicy` problems (DTensor RNG tracker on CPU tensors; `trunc_normal_` hang on 2.13).

## 9. Ownership recommendation

Platform stack (PyTorch XPU / XCCL / oneCCL), not ROLL. File in **intel/torch-xpu-ops** (where the FSDP XPU hangs are tracked) with the bundle `fsdp2_xpu_cpu_offload_broadcast_hang/`; cross-reference oneCCL only if the collective layer is identified as the failing component.

## 10. Bug-report link

Not filed yet (to be filed by the owner of the GitHub account). Bundle ready: `fsdp2_xpu_cpu_offload_broadcast_hang/` (README.md, repro.py, run.sh, expected.txt, actual.txt, collect_env.txt, pip_freeze.txt, system_packages.txt, rank0.log, rank1.log, stack_rank0.txt, stack_rank1.txt, model_matrix.md).

## 11. ROLL status

| Test | Status |
|---|---|
| `test_fsdp_log_probs_lora`, 2 XPUs | **clean pass**, 3 of 3 (only `device_mapping` and `fsdp_size` changed) |
| `test_fsdp_log_probs_full`, 2 XPUs, original CPU offload | **platform-stack blocked** (hang 2 of 2; ROLL-free reproducer hang 3 of 3) |
| `full` with offload disabled | diagnostic pass only; not the original test |

Not a ROLL failure. The offload-disabled run is not presented as the original test passing.

## 12. CRI

Not run (team cap 4 CRI cards, node availability). Plan if 2 CRI GPUs become available: the minimal reproducer only, 3 repetitions, same conditions; interpretation per the plan (CRI pass + B60 fail → BMG/driver/Level Zero path; both fail → shared PyTorch FSDP2/XCCL path).
