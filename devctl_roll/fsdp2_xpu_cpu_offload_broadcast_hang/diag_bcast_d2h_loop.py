"""Minimal pattern test, no FSDP, no state dict: N times { XCCL broadcast of a small bf16 tensor from rank 0; copy a slice of it to CPU }.
usage: diag_bcast_d2h_loop.py <rank> <world> <port> <iters> [sync|nosync]  (sync = D2H copy after every broadcast, like the CPU-offload path)"""
import os, sys, time
import torch, torch.distributed as dist
rank, world, port, iters = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
d2h = (sys.argv[5] if len(sys.argv) > 5 else "sync") == "sync"
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port, RANK=str(rank), WORLD_SIZE=str(world))
torch.xpu.set_device(0)
dist.init_process_group(backend="cpu:gloo,xpu:xccl", device_id=torch.device("xpu", 0))
t0 = time.time()
shapes = [(1536, 1536), (256, 1536), (256, 1536), (1536, 1536), (8960, 1536), (8960, 1536), (1536, 8960), (1536,), (1536,)]  # one Qwen2.5-1.5B layer
for i in range(iters):
    s = shapes[i % len(shapes)]
    x = torch.randn(s, dtype=torch.bfloat16, device="xpu") if rank == 0 else torch.empty(s, dtype=torch.bfloat16, device="xpu")
    dist.broadcast(x, src=0)
    if d2h:
        half = x.shape[0] // world
        cpu = torch.empty((half,) + tuple(x.shape[1:]), dtype=torch.bfloat16)
        cpu.copy_(x[rank * half:(rank + 1) * half])
    if i % 50 == 0:
        print(f"rank {rank} {time.strftime('%H:%M:%S')} iter {i} ok {time.time()-t0:.1f}s", flush=True)
torch.xpu.synchronize(); dist.barrier()
print(f"rank {rank} LOOP_OK iters={iters} d2h={d2h} {time.time()-t0:.1f}s", flush=True)
dist.destroy_process_group()
