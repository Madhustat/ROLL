# ROLL validation summary (context for the bug report)

ROLL = alibaba/ROLL `d3506ea9` + Intel XPU enablement (branch `xpu/roll-xpu-enablement`, Madhustat/ROLL). Tests from `tests/distributed/strategy/log_probs/test_fsdp_log_probs.py`. Runs on local 2 × Intel Arc Pro B70 with the upstream code paths (no local workaround variables), one process per GPU, oneCCL 2022.0.0, PyTorch 2.14.0+xpu.

| Test | Change made | Result | Classification |
|---|---|---|---|
| `test_fsdp_log_probs_lora` | `device_mapping` 0-7 → 0-1 (two roles), `fsdp_size` 8 → 2 | **pass 3 of 3** (101 s, 82 s, 82 s; identical metrics: max per-token diff vs reference 0.612, row-mean max 0.034) | clean pass (only device count changed) |
| `test_fsdp_log_probs_full` (config has `offload_policy: true`) | same device-count change only | **hang 2 of 2** at the FSDP2 start-up weight broadcast (also hung in about 5 of 5 start-ups on 8 × Arc Pro B60 on devctl) | platform-stack blocked |
| `test_fsdp_log_probs_full` with `offload_policy: False` | functional change for diagnosis only | pass (72 s) | diagnostic pass only, **not** the original test |
| `test_fsdp_log_probs_cp`, `test_fsdp_log_probs_cp_rmpad` | none | not run | resource-blocked: 8 devices required by code and topology, and the unchanged single-device reference path needs more than about 34 GB per GPU (measured out of memory on 32 GB) |

Rules kept throughout: no ROLL Python source changed, no test Python file changed, `offload_policy` unchanged for the reported result, `ROLL_FSDP2_INIT_WITHOUT_BROADCAST` and `ROLL_LOGITS_CHUNK_SIZE` set to `0` in every run.

The ROLL hang and the PyTorch-only reproducer stop at the same frames (`torch/distributed/_state_dict_utils.py:636 ← 595 ← 681 ← checkpoint/state_dict.py:609 ← set_model_state_dict`), called from ROLL's `roll/utils/fsdp_utils.py:fsdp2_load_full_state_dict`.

Earlier result that looked contradictory: a ROLL pipeline with Qwen2.5-1.5B and `offload_policy: true` passed on 2026-11-04. Its policy ran on **one** device (`device_mapping: list(range(0,1))`), so `set_model_state_dict(broadcast_from_rank0=True)` involved no inter-rank broadcast; the failing condition was not exercised.
