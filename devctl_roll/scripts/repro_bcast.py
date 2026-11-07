"""N-rank reproducer of ROLL's FSDP2 startup broadcast (set_model_state_dict broadcast_from_rank0). usage: repro_bcast.py <rank> <world> <port> <variant>
variants (combine with +): roll, warmup (1-element all_reduce first, as ROLL's train strategy), setdev (set_device + device_id before init);
algo and nothresh are applied by run_repro.sh through the environment."""
import os, sys, time
import torch
import torch.distributed as dist

rank, world, port, variant = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port, RANK=str(rank), WORLD_SIZE=str(world))
t0 = time.time()
if "setdev" in variant:
    torch.xpu.set_device(0)
    dist.init_process_group(backend="cpu:gloo,xpu:xccl", device_id=torch.device("xpu", 0))
else:
    dist.init_process_group(backend="cpu:gloo,xpu:xccl")
if "warmup" in variant:
    dist.all_reduce(torch.zeros(1).to("xpu"))
shapes = [(151936, 1024)] + [(1024,), (2048, 1024), (1024, 1024), (1024, 1024), (1024, 2048), (3072, 1024), (3072, 1024), (1024, 3072), (1024,)] * 28
tensors = [torch.empty(s, dtype=torch.bfloat16, device="xpu") for s in shapes]
if rank == 0:
    for t in tensors:
        t.normal_()
dist._broadcast_coalesced(dist.group.WORLD, tensors, 500 * 1024 * 1024, 0)
torch.xpu.synchronize()
chk = torch.stack([t.float().sum() for t in tensors]).sum().reshape(1)
ref = chk.clone()
dist.broadcast(ref, src=0)
torch.xpu.synchronize()
ok = bool(torch.allclose(chk, ref))
dist.barrier()
print(f"rank {rank}/{world} variant {variant} BROADCAST_OK data_ok={ok} in {time.time() - t0:.1f}s", flush=True)
dist.destroy_process_group()
