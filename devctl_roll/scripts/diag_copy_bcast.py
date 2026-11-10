"""Synthetic diagnostic, no ROLL: per step, synchronize + barrier. usage: diag_copy_bcast.py <rank> <world> <port>
Sizes match the FSDP start-up tensors of Qwen3-0.6B: 311 MB (embedding), 6 MB and 2 MB (layer weights)."""
import os, sys, time
import torch, torch.distributed as dist
rank, world, port = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port, RANK=str(rank), WORLD_SIZE=str(world))
torch.xpu.set_device(0)
dist.init_process_group(backend="cpu:gloo,xpu:xccl", device_id=torch.device("xpu", 0))
t0 = time.time()
def step(name):
    torch.xpu.synchronize(); dist.barrier()
    print(f"rank {rank} {time.strftime('%H:%M:%S')} OK {name} {time.time()-t0:.1f}s", flush=True)
for n in (151936 * 1024, 3072 * 1024, 1024 * 1024):
    mb = n * 2 // 2**20
    cpu = torch.randn(n, dtype=torch.bfloat16)
    x = cpu.to("xpu"); step(f"pageable_H2D_{mb}MB")
    pin = cpu.pin_memory(); x = pin.to("xpu", non_blocking=True); step(f"pinned_H2D_{mb}MB")
    dist.broadcast(x, src=0); step(f"xccl_bcast_{mb}MB")
    out = torch.empty(n // 2, dtype=torch.bfloat16); out.copy_(x[: n // 2]); step(f"bcast_then_D2H_slice_pageable_{mb}MB")
    outp = torch.empty(n // 2, dtype=torch.bfloat16).pin_memory(); outp.copy_(x[: n // 2]); step(f"bcast_then_D2H_slice_pinned_{mb}MB")
    y = torch.empty_like(x); dist.broadcast(y, src=0); half = y[n // 2 :]; outc = torch.empty(n - n // 2, dtype=torch.bfloat16); outc.copy_(half); step(f"bcast_then_D2H_second_half_{mb}MB")
print(f"rank {rank} DIAG_OK {time.time()-t0:.1f}s", flush=True)
dist.destroy_process_group()
