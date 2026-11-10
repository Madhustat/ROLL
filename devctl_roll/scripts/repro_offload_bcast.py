"""ROLL-free: FSDP2 CPUOffloadPolicy + set_model_state_dict(broadcast_from_rank0=True, cpu_offload=True) on Qwen3-0.6B, 2 ranks.
usage: repro_offload_bcast.py <rank> <world> <port> <on|off> [bcast|nobcast] [model]
nobcast = every rank keeps its CPU checkpoint and set_model_state_dict runs with broadcast_from_rank0=False (local distribute)."""
import faulthandler, os, sys, time
import torch, torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from transformers import AutoModelForCausalLM
rank, world, port, off = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4] == "on"
bcast = (sys.argv[5] if len(sys.argv) > 5 else "bcast") == "bcast"
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port, RANK=str(rank), WORLD_SIZE=str(world))
t0 = time.time(); faulthandler.dump_traceback_later(200, exit=False)
mark = lambda m: print(f"rank {rank} {time.strftime('%H:%M:%S')} STAGE {m} {time.time()-t0:.1f}s xpu_alloc_MiB={torch.xpu.memory_allocated()//2**20 if torch.xpu.is_initialized() else 0}", flush=True)
print(f"rank {rank} pid={os.getpid()} ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK')} device={torch.xpu.get_device_name(0)} start={time.strftime('%H:%M:%S')}", flush=True)
torch.xpu.set_device(0)
dist.init_process_group(backend="cpu:gloo,xpu:xccl", device_id=torch.device("xpu", 0)); mark("init")
mesh = init_device_mesh("xpu", (world,))
mark("before_model_construct")
model = AutoModelForCausalLM.from_pretrained(sys.argv[6] if len(sys.argv) > 6 else "Qwen/Qwen3-0.6B", dtype=torch.bfloat16); mark("after_model_construct")
full_state = torch.nn.Module.state_dict(model)
pol = CPUOffloadPolicy() if off else None
mark("before_fully_shard")
for layer in model.model.layers: fully_shard(layer, mesh=mesh, offload_policy=pol)
fully_shard(model, mesh=mesh, offload_policy=pol); mark("after_fully_shard")
model = model.to("xpu") if (rank == 0 or not bcast) else model.to_empty(device="xpu")
mark("before_set_model_state_dict")
set_model_state_dict(model, full_state, options=StateDictOptions(full_state_dict=True, cpu_offload=off, broadcast_from_rank0=bcast)); mark("after_set_model_state_dict")
torch.xpu.synchronize(); mark("after_synchronize")
dist.barrier(); mark("after_barrier")
print(f"rank {rank} offload={'on' if off else 'off'} bcast={bcast} LOAD_OK {time.time()-t0:.1f}s", flush=True)
dist.destroy_process_group()
