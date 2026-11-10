"""Minimal reproducer: FSDP2 CPUOffloadPolicy + set_model_state_dict(broadcast_from_rank0=True) hangs on Intel XPU / XCCL.
No ROLL, no Ray. Run one process per GPU (see run.sh).

usage: repro.py --rank R --world N --port P [--offload on|off] [--broadcast on|off] [--model PATH]
"""
import argparse, faulthandler, os, time
import torch, torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from transformers import AutoModelForCausalLM

a = argparse.ArgumentParser()
a.add_argument("--rank", type=int, required=True); a.add_argument("--world", type=int, default=2); a.add_argument("--port", default="29600")
a.add_argument("--offload", choices=["on", "off"], default="on"); a.add_argument("--broadcast", choices=["on", "off"], default="on")
a.add_argument("--model", default="Qwen/Qwen3-0.6B"); a.add_argument("--dump-after", type=int, default=120)
args = a.parse_args()
rank, world, off, bcast = args.rank, args.world, args.offload == "on", args.broadcast == "on"
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=args.port, RANK=str(rank), WORLD_SIZE=str(world))
t0 = time.time(); faulthandler.dump_traceback_later(args.dump_after, exit=False)
def mark(m):
    alloc = torch.xpu.memory_allocated() // 2**20 if torch.xpu.is_initialized() else 0
    print(f"rank {rank} {time.strftime('%H:%M:%S')} STAGE {m} {time.time()-t0:.1f}s xpu_alloc_MiB={alloc}", flush=True)
print(f"rank {rank} pid={os.getpid()} ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK')} device={torch.xpu.get_device_name(0)}", flush=True)
torch.xpu.set_device(0)
dist.init_process_group(backend="cpu:gloo,xpu:xccl", device_id=torch.device("xpu", 0)); mark("init_process_group")
mesh = init_device_mesh("xpu", (world,))
mark("before_model_construct")
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16); mark("after_model_construct")
full_state = torch.nn.Module.state_dict(model)
shared = sum(1 for v in full_state.values() if v.untyped_storage().data_ptr() == full_state.get("lm_head.weight", torch.empty(0)).untyped_storage().data_ptr()) if "lm_head.weight" in full_state else 0
print(f"rank {rank} params={sum(p.numel() for p in model.parameters())} tensors={len(full_state)} bytes={sum(v.numel()*v.element_size() for v in full_state.values())} tied={getattr(model.config,'tie_word_embeddings',None)} shared_storage_keys={shared}", flush=True)
pol = CPUOffloadPolicy() if off else None
mark("before_fully_shard")
for layer in model.model.layers: fully_shard(layer, mesh=mesh, offload_policy=pol)
fully_shard(model, mesh=mesh, offload_policy=pol); mark("after_fully_shard")
model = model.to("xpu") if (rank == 0 or not bcast) else model.to_empty(device="xpu")
mark("before_set_model_state_dict")
set_model_state_dict(model, full_state, options=StateDictOptions(full_state_dict=True, cpu_offload=off, broadcast_from_rank0=bcast)); mark("after_set_model_state_dict")
torch.xpu.synchronize(); mark("after_synchronize")
dist.barrier(); mark("after_barrier")
print(f"rank {rank} offload={args.offload} broadcast={args.broadcast} LOAD_OK {time.time()-t0:.1f}s", flush=True)
dist.destroy_process_group()
