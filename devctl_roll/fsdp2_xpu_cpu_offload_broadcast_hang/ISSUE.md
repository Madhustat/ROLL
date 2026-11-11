Title: [FSDP2][XPU][Regression?] set_model_state_dict with CPUOffloadPolicy and broadcast_from_rank0=True hangs over XCCL on torch 2.14 (possible regression of #4238, closed as fixed in 2.13)

## Summary

Loading an FSDP2-sharded model with `fully_shard(..., offload_policy=CPUOffloadPolicy())` and then `set_model_state_dict(model, full_state, StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True))` hangs on Intel XPU over XCCL with 2 ranks. Deterministic: 3 of 3 runs. PyTorch only; no ROLL, no Ray. The same model and the same collective complete when either CPU offload or the rank-0 broadcast is removed. This closely matches a previously-reported, **closed** issue on the same hardware class (#4238, "fixed in PyTorch 2.13") — we reproduce the identical symptom on PyTorch 2.14, which suggests a regression rather than a new defect.

## Environment

- Hardware: 2 × Intel Arc Pro B70 (32 GB) for the attached reproducer; the identical hang (same stack) is seen on 8 × Intel Arc Pro B60 (24 GB) through a framework's FSDP2 start-up with the same software stack.
- World size: 2, one process per GPU (`ZE_AFFINITY_MASK=0` / `1`).
- PyTorch 2.14.0+xpu, git `08187d9e0fba`; XCCL backend (built in); `init_process_group(backend="cpu:gloo,xpu:xccl", device_id=xpu:0)`.
- oneCCL runtime 2022.0.0 (preloaded); intel-sycl-rt 2026.1.1; Level Zero loader 1.32; NEO 26.31.39395.14 (driver `1.17.39395+14`); Ubuntu 24.04.2, kernel 7.0.0-42. `collect_env.txt`, `pip_freeze.txt`, `system_packages.txt` attached.
- Model: `Qwen/Qwen3-0.6B`, bf16 (also reproduces with `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct`).

## Reproducer

`repro.py` / `run.sh` attached. Steps: all ranks load the checkpoint on CPU in bf16; `fully_shard` per decoder layer and root with `CPUOffloadPolicy()`; rank 0 `model.to("xpu")`, rank 1 `model.to_empty(device="xpu")`; `set_model_state_dict(..., full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True)`; `torch.xpu.synchronize()`; `dist.barrier()`.

```
./run.sh on  on    # CPU offload on,  broadcast on  -> hang 3/3 (killed at 300 s)
./run.sh off on    # CPU offload off, broadcast on  -> pass 3/3 (~2 s)
./run.sh on  off   # CPU offload on,  broadcast off -> pass 3/3 (~3 s)
./run.sh off off   # pass
```

## Observed

Both ranks print `STAGE before_set_model_state_dict` about 2 s after start and never print `after_set_model_state_dict`. Python stacks of both ranks (identical):

```
_distribute_tensors    torch/distributed/_state_dict_utils.py:636   # ret.to_local().copy_(full_tensor[slices])
_broadcast_tensors     torch/distributed/_state_dict_utils.py:595
_broadcast_state_dict  torch/distributed/_state_dict_utils.py:681
_load_model_state_dict torch/distributed/checkpoint/state_dict.py:609
set_model_state_dict   torch/distributed/checkpoint/state_dict.py:1435
```

The wait is in the per-tensor copy of the broadcast result into the CPU-offloaded local shard, after `_broadcast_coalesced` has returned; not in `torch.xpu.synchronize()`. Live `py-spy dump --locals` during a hang (Qwen2.5-1.5B) shows the two ranks on **different tensors**: rank 1 on `model.layers.0.self_attn.k_proj.weight` (offset 128), rank 0 already on `model.layers.0.self_attn.v_proj.weight`. `_broadcast_tensors` is entered with one key at a time (500-byte coalescing limit), so each tensor is one `dist.broadcast` followed by a device-to-host copy. No XCCL error or timeout. Kernel log shows no GPU fault during the hang; engine resets only appear when the hung processes are killed.

## Controls that pass

- CPU offload disabled (same broadcast): 3 of 3.
- Rank-0 broadcast disabled (every rank loads its checkpoint, local distribute): 3 of 3.
- Plain XCCL broadcasts (296 MB / 6 MB / 2 MB), pageable and pinned host-to-device copies, broadcast followed by device-to-host slice copies: pass.
- 400 × {small XCCL broadcast; device-to-host slice copy}, with and without the host copy: pass.
- Environment settings tried one at a time on the failing case, all still hang: `TORCH_XCCL_BLOCKING_WAIT=1`, `CCL_SYCL_OUTPUT_EVENT=1`, `CCL_BCAST=ring`, `CCL_ZE_IPC_EXCHANGE=sockets`, `CCL_ZE_COPY_ENGINE=none`, `CCL_ZE_DEPS_SYNC=1`, `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0`. `CCL_ATL_TRANSPORT=ofi` / `FI_PROVIDER=tcp` are already the defaults. No `PYTORCH_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF` set.

## Model matrix (offload on, broadcast on)

| Model | Params | Tied embeddings | Tensors | Result |
|---|---|---|---|---|
| Qwen3-0.6B | 752 M | yes | 311 | hang 3/3 |
| Qwen2.5-0.5B-Instruct | 494 M | yes | 290 | hang |
| Qwen2.5-1.5B-Instruct | 1544 M | yes | 338 | hang |

An earlier run of a Qwen2.5-1.5B pipeline with CPU offload that passed used a single FSDP2 rank, so it executed no inter-rank broadcast and is not a counter-example.

## Related public issues (closest match leads; not claimed to be confirmed duplicates)

- **intel/torch-xpu-ops #4238** (closed, "fixed in PyTorch 2.13"): identical repro shape to ours — FSDP `sync_module_states`/`set_model_state_dict` hangs at the init broadcast with CPU offload enabled, on the **same hardware class (2x Arc Pro B70)**, stalling inside a Level Zero driver memcpy during the broadcast. The reporter confirmed upgrading from torch 2.12 to 2.13 resolved it with no settings changed. We are on **torch 2.14** and hit the identical symptom — this looks like a **regression** of a previously-fixed issue, not a brand-new bug.
- intel/torch-xpu-ops #5999, #5968, #6092: related FSDP/XPU hang reports (checkpoint, grad-scaler, unshard); #5968 traces one case to a Level Zero adapter timing bug, fix pending a oneAPI pin bump.
- pytorch/pytorch #202028, #196562: a different FSDP2 `CPUOffloadPolicy` bug (DTensor RNG tracker resets CPU randomness; not a hang).


## Attachments

`repro.py`, `run.sh`, `expected.txt`, `actual.txt`, `collect_env.txt`, `pip_freeze.txt`, `system_packages.txt`, `logs/` (3 failing runs + 0.5B/1.5B runs, 3 offload-off controls, 3 broadcast-off controls, per-rank, with timestamped progress markers), `logs/stacks/` (faulthandler and `py-spy --locals` dumps of both ranks), `model_matrix.md`, `ROLL_VALIDATION.md`.
