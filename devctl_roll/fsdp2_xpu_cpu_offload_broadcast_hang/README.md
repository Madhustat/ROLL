# FSDP2 `CPUOffloadPolicy` + `set_model_state_dict(broadcast_from_rank0=True)` hangs on Intel XPU (XCCL), 2 ranks

Deterministic hang, reproduced without ROLL and without Ray. The same model and the same collective complete when **either** CPU offload **or** the rank-0 broadcast is removed.

## Environment

| Item | Value |
|---|---|
| Hardware (this bundle) | 2 × Intel Arc Pro B70 (32 GB), one process per GPU (`ZE_AFFINITY_MASK=0` / `1`) |
| Also seen on | 8 × Intel Arc Pro B60 (24 GB) through ROLL's FSDP2 start-up (same stack, Kubernetes pod, NEO 26.14) |
| PyTorch | 2.14.0+xpu, git `08187d9e0fba` (`collect_env.txt`) |
| XCCL / oneCCL | XCCL backend built into PyTorch (no `oneccl-bind-pt`); oneCCL runtime **2022.0.0** preloaded (`LD_PRELOAD` of `libccl.so.2`; the pip `oneccl` package in the venv is 2022.1.2 and is overridden) |
| SYCL runtime | `intel-sycl-rt 2026.1.1`, `dpcpp-cpp-rt 2026.1.1`, `umf 1.1.0` |
| Level Zero loader / GPU driver | `libze1` 1.32, NEO (`libze-intel-gpu1`, `intel-opencl-icd`) 26.31.39395.14, driver string `1.17.39395+14` (`system_packages.txt`) |
| OS / kernel | Ubuntu 24.04.2, 7.0.0-42-generic |
| Model | `Qwen/Qwen3-0.6B`, bf16 (also hangs with `Qwen/Qwen2.5-0.5B-Instruct`; see `model_matrix.md`) |
| Process group | `init_process_group(backend="cpu:gloo,xpu:xccl", device_id=xpu:0)`, `init_device_mesh("xpu", (2,))` |

## Command

```bash
./run.sh on on            # CPU offload on,  rank-0 broadcast on  -> HANG (3 of 3)
./run.sh off on           # CPU offload off, rank-0 broadcast on  -> pass (3 of 3, ~2 s)
./run.sh on off           # CPU offload on,  broadcast off (each rank loads) -> pass (3 of 3, ~3 s)
./run.sh off off          # control -> pass
```

`run.sh` starts two ranks with a 300 s limit; `repro.py` prints a timestamped marker before/after every step and dumps all Python stacks at 120 s (`faulthandler`).

## What the reproducer does (mirrors FSDP2 start-up in ROLL and similar frameworks)

1. All ranks load the checkpoint on CPU in bf16 (`AutoModelForCausalLM.from_pretrained`) and take `full_state = nn.Module.state_dict(model)`.
2. `fully_shard` per decoder layer and on the root, `offload_policy=CPUOffloadPolicy()` (or `None`).
3. Rank 0: `model.to("xpu")`; other ranks: `model.to_empty(device="xpu")`.
4. `set_model_state_dict(model, full_state, StateDictOptions(full_state_dict=True, cpu_offload=<offload>, broadcast_from_rank0=True))`.
5. `torch.xpu.synchronize()`, `dist.barrier()`.

## Results

| Condition | Runs | Result |
|---|---|---|
| offload on, broadcast on | 3 | **hang, 3 of 3** (killed at 300 s) |
| offload off, broadcast on | 3 | pass |
| offload on, broadcast off | 3 | pass |
| offload off, broadcast off | 1 | pass |

Last progress marker on **both** ranks: `STAGE before_set_model_state_dict` at about 2 s. `after_set_model_state_dict` is never printed.

Where the ranks wait (identical on both ranks, `stack_rank0.txt` / `stack_rank1.txt`):

```
_distribute_tensors   torch/distributed/_state_dict_utils.py:636   # ret.to_local().copy_(full_tensor[slices])
_broadcast_tensors    torch/distributed/_state_dict_utils.py:595   # after dist._broadcast_coalesced(pg, tensors, 500, 0)
_broadcast_state_dict torch/distributed/_state_dict_utils.py:681
_load_model_state_dict torch/distributed/checkpoint/state_dict.py:609
set_model_state_dict  torch/distributed/checkpoint/state_dict.py:1435
```

So the process is not inside `torch.xpu.synchronize()`; it is in the per-tensor copy of the broadcast result into the (CPU-offloaded) local shard, after `_broadcast_coalesced` has returned. No error, no timeout from XCCL. The kernel log shows **no GPU fault** during the hang; engine resets appear only when the hung processes are killed.

Live stacks captured during the Qwen2.5-1.5B hang (`py-spy dump --locals`, both ranks in `_distribute_tensors`): **the ranks are on different tensors**: rank 1 is copying its shard of `model.layers.0.self_attn.k_proj.weight` (shape (128, 1536), offset (128, 0)), rank 0 is already on `model.layers.0.self_attn.v_proj.weight` (offset (0, 0)). `_broadcast_tensors` is entered with a single key each time (`keys: ["model.layers.0.self_attn.k_proj.weight"]`), so each tensor is one `dist.broadcast` followed, with CPU offload, by a device-to-host copy on every rank. The embedding (445 MiB) and `layers.0.q_proj` completed before the ranks fell out of step.

Note on bucket size: `_broadcast_tensors` calls `dist._broadcast_coalesced(pg, tensors, 500, 0)`; with a 500-**byte** limit no two tensors coalesce, so the state dict is sent as one broadcast per tensor (311 for Qwen3-0.6B, 1.40 GiB in total).

## Settings tried (one at a time, on the failing condition; none avoids the hang)

`TORCH_XCCL_BLOCKING_WAIT=1`, `CCL_SYCL_OUTPUT_EVENT=1`, `CCL_BCAST=ring`, `CCL_ZE_IPC_EXCHANGE=sockets`, `CCL_ZE_COPY_ENGINE=none`, `CCL_ZE_DEPS_SYNC=1`: all hang. `CCL_ZE_ENABLE=0`: XCCL cannot create a communicator. `CCL_ATL_TRANSPORT=ofi` / `FI_PROVIDER=tcp`: already the defaults (oneCCL log: `provider: tcp`, `libfabric 2.4.0-impi`). `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0`: hang. No `PYTORCH_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF` is set in the environment.

## Additional passing control

`diag_bcast_d2h_loop.py` (in the same clean2xpu folder): 400 × {`dist.broadcast` of a small bf16 tensor; copy of a slice to CPU}, with and without the host copy: passes (4 s / 3 s). The hang therefore needs the FSDP2 state-dict path (`CPUOffloadPolicy` DTensor shards + `set_model_state_dict`), not merely a broadcast followed by a host sync.

## Model dependence (`model_matrix.md`)

| Model | Offload on + broadcast on |
|---|---|
| Qwen3-0.6B (752 M, tied) | hang 3 of 3 |
| Qwen2.5-0.5B-Instruct (494 M, tied) | hang (1 run) |
| Qwen2.5-1.5B-Instruct (1544 M, tied) | hang (1 run, 300 s; live stacks captured, see below) |

## Related public issues (related evidence, not claimed to be the same root cause)

- intel/torch-xpu-ops #5999: FSDP tests hang until the 300 s distributed-harness timeout (marked duplicate of #5968).
- intel/torch-xpu-ops #5968: xpu_distributed FSDP checkpoint / grad-scaler / unshard cases time out; label `dependency component: Level_Zero`; traced to a UR Level Zero adapter abort fixed upstream, awaiting a oneAPI pin bump.
- intel/torch-xpu-ops #6092: FSDP unshard and backward-reshard multi-process tests hang (regression).
- pytorch/pytorch #202028 and #196562: FSDP2 `CPUOffloadPolicy` correctness problems (DTensor RNG tracker applied to CPU tensors; `trunc_normal_` hang on 2.13).

## Files

`repro.py`, `run.sh`, `expected.txt`, `actual.txt`, `collect_env.txt`, `pip_freeze.txt`, `system_packages.txt`, `rank0.log`, `rank1.log`, `stack_rank0.txt`, `stack_rank1.txt`, `model_matrix.md`.
