# devctl Setup: ROLL (RL post-training) on Intel XPU

Runs ROLL's GRPO recipe (Qwen2.5-0.5B-Instruct) on Intel GPUs through devctl.
On bare metal (2 × Arc Pro B70) this exact stack and patch pass:
- 1-GPU colocated and 2-GPU disaggregated GRPO;
- bit-exact weight sync.

**Not yet run on devctl / B60.**

Stack:
- torch 2.14.0+xpu, vLLM 0.31.0+xpu, Ray 2.48.0, Transformers 5.17.0;
- oneCCL 2022.0.0 (preloaded; the venv's default 2022.1.2 must not be used);
- ROLL commit d3506ea9 + `roll_xpu_d3506ea9.patch` (XPU platform file + 7 small edits).

Layout inside every devctl job:
- `/workspace/local`: synced from `~/src/roll-devctl` (ROLL + this folder).
- `/workspace/remote`: persistent; holds the venv, oneCCL 2022.0.0, models, outputs.

Full commands: see the "All commands" section.

On branch `xpu/devctl-roll-setup` of github.com/Madhustat/ROLL the patch is already applied. Skip the scp and `git apply` lines of step 1; instead use:
`git fetch origin xpu/devctl-roll-setup && git checkout xpu/devctl-roll-setup && cp -r devctl_roll ..`

## All commands (run from WSL)

```bash
# 0) Prerequisite: devctl built, on PATH, and ~/.devctl/config.yaml pointing at the cluster kubeconfig (same as the Unsloth guide).
devctl --help

# 1) Bring the bundle and ROLL into WSL (keep it under /home, not /mnt/c)
mkdir -p ~/src/roll-devctl && cd ~/src/roll-devctl
scp sdp@spr818157.jf.intel.com:/madhu_Roll/devctl_roll.tar.gz . && tar xzf devctl_roll.tar.gz
git clone https://github.com/Madhustat/ROLL.git ROLL
cd ROLL && git checkout d3506ea9 && git apply ../devctl_roll/roll_xpu_d3506ea9.patch && git status --short && cd ..

# 2) Session for ROLL (100Gi: venv ~10 GB + models + HF cache + outputs) and first sync
devctl session create roll-xpu --workspace=$HOME/src/roll-devctl --size=100Gi
devctl session current
devctl session sync

# 3) One-time environment build in /workspace/remote (CPU job, ~20-40 min). Expect "torch 2.14.0+xpu xpu ..." and "SETUP DONE".
devctl test --skip-sync --image=intel/pytorch:xpu-2.11.0-ubuntu24.04 --gpu=0 --cpu=8 --memory=32Gi \
  --command="bash,/workspace/local/devctl_roll/scripts/setup_env.sh" --attach

# 4) Download the model (CPU job). Expect "DOWNLOAD DONE".
devctl test --skip-sync --image=intel/pytorch:xpu-2.11.0-ubuntu24.04 --gpu=0 --cpu=4 --memory=8Gi \
  --command="bash,/workspace/local/devctl_roll/scripts/download_model.sh,Qwen/Qwen2.5-0.5B-Instruct" --attach

# 5) Smoke test on 1 GPU. Expect: HEALTH OK, "ROLL platform: XpuPlatform | backend: xccl | supports_tensor_ipc: False",
#    dpctl level_zero gpus: 1, LD_PRELOAD .../oneccl_2022_0_0/lib/libccl.so.2, "SMOKE DONE".
devctl test --skip-sync --image=intel/pytorch:xpu-2.11.0-ubuntu24.04 --gpu=1 --gpu-model=B60 --cpu=4 --memory=16Gi \
  --command="bash,/workspace/local/devctl_roll/scripts/smoke.sh" --attach

# 6) GRPO on 1 GPU (training, reference and vLLM colocated; weight sync via oneCCL same-GPU IPC). Expect "pipeline complete!" and "RUN DONE rc=0".
devctl test --skip-sync --image=intel/pytorch:xpu-2.11.0-ubuntu24.04 --gpu=1 --gpu-model=B60 --cpu=8 --memory=64Gi \
  --command="bash,/workspace/local/devctl_roll/scripts/run_roll.sh,start_rlvr_pipeline.py,grpo_qwen25_05b_1xpu_colocated" --attach

# 7) GRPO on 2 GPUs (training + reference on GPU 0, vLLM on GPU 1; XPU-to-XPU weight broadcast over XCCL)
devctl test --skip-sync --image=intel/pytorch:xpu-2.11.0-ubuntu24.04 --gpu=2 --gpu-model=B60 --cpu=8 --memory=64Gi \
  --command="bash,/workspace/local/devctl_roll/scripts/run_roll.sh,start_rlvr_pipeline.py,grpo_qwen25_05b_2xpu_disagg" --attach

# 8) Monitor / outputs
devctl list
devctl status <workload-name>
devctl logs <workload-name> --tail=100
# Outputs persist in /workspace/remote/roll_work/{output,checkpoints,tensorboard}

# 9) If a run fails with an IPC error from oneCCL (e.g. pidfd_getfd "Operation not permitted"), re-run step 6/7 adding:
#      --env=ROLL_CCL_IPC=sockets
# After editing anything under ~/src/roll-devctl: devctl session sync (or drop --skip-sync).
```

## Not yet verified on devctl (check these first if a step fails)
| Risk | Symptom | What to do |
|---|---|---|
| Image lacks `python3.12-venv` | setup stops at the venv check | setup tries `apt-get install python3.12-venv` (needs root in the pod) |
| Pod forbids `pidfd_getfd` (needs ptrace capability; on bare-metal Docker we needed SYS_PTRACE) | oneCCL IPC error in step 6/7 | `--env=ROLL_CCL_IPC=sockets` |
| Small `/dev/shm` in pods (Kubernetes default 64 MB) | Ray warns "/dev/shm too small, using /tmp" | runs, slower; ask infra for a larger shm if needed |
| B60 has 24 GB (bare-metal tests used 32 GB B70) | out of memory on vLLM start | lower `gpu_memory_utilization` in the config from 0.4 to 0.3 |
| No oneAPI compiler (`icpx`) in the image | Triton kernel build error (`CL/cl.h` not found) | use an image with oneAPI, or install the oneAPI DPC++ compiler into /workspace/remote |

## Unit tests on 8 GPUs

```bash
# GPU tests that skipped on 2 XPUs (log-probs tests place workers on devices 0-7) + the other multi-GPU tests.
# Expect a PASSED/FAILED/SKIPPED count and "UNIT TESTS DONE rc=...". Log + junit in /workspace/remote/unit_tests/<time>_gpu/
devctl test --skip-sync --image=intel/pytorch:xpu-2.11.0-ubuntu24.04 --gpu=8 --gpu-model=B60 --cpu=16 --memory=128Gi \
  --command="bash,/workspace/local/devctl_roll/scripts/run_unit_tests.sh,gpu" --attach

# Full ROLL test suite on the same 8 GPUs (long; same output location, suffix _all)
devctl test --skip-sync --image=intel/pytorch:xpu-2.11.0-ubuntu24.04 --gpu=8 --gpu-model=B60 --cpu=16 --memory=128Gi \
  --command="bash,/workspace/local/devctl_roll/scripts/run_unit_tests.sh,all" --attach

# If the 8-GPU job stays Pending (no free 8-GPU node), check availability:
kubectl --kubeconfig=$HOME/.kube/<your>.kubeconfig get resourceclaims -A
```

Notes:
- `test_hf_multi_gpus.py::test_hf_multi_gpus_cpu_offload_hf_device_map` is deselected: it loops over all 36,441 data rows (~15 h). ROLL's own CI also skips it on NPU.
- `test_ray_thread_actor_cuda_mem_leak.py` allocates 39 GiB per call, so it will still fail on 24 GB B60 (it failed on 32 GB B70 too). It needs a larger GPU.
- Tests download their models from Hugging Face (`MODEL_DOWNLOAD_TYPE=HUGGINGFACE_HUB`); the pod needs HF access.
