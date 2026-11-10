# Model matrix (static properties from the checkpoint files; bucket partition = greedy fill of PyTorch's 500-byte coalescing limit in state-dict key order)

| Model | Params | Tied embeddings | Tensors | Largest tensor | Total bf16 bytes | Buckets (500 B limit) | Tensors in multi-tensor buckets | Separate lm_head key |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 752 M | True | 311 | `lm_head.weight` (151936, 1024) = 297 MiB | 1.40 GiB | 311 | 0 | True |
| Qwen2.5-0.5B-Instruct | 494 M | True | 290 | `model.embed_tokens.weight` (151936, 896) = 260 MiB | 0.92 GiB | 290 | 0 | False |
| Qwen2.5-1.5B-Instruct | 1544 M | True | 338 | `model.embed_tokens.weight` (151936, 1536) = 445 MiB | 2.88 GiB | 338 | 0 | False |

Notes: parameter count is from the checkpoint tensors (tied models store the embedding once, Qwen3-0.6B stores both `embed_tokens` and `lm_head`). Bucket partition is computed from the checkpoint key order; PyTorch's `_broadcast_coalesced(..., 500, 0)` coalesces tensors only while a bucket stays within 500 bytes, so every tensor larger than 500 bytes is its own bucket.
