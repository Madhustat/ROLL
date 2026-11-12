import os
import socket

import torch

from .platform import Platform
from ..utils.logging import get_logger


logger = get_logger()

ASSIGNED_XPU_IDS_ENV = "ROLL_ASSIGNED_XPU_IDS"
PRIMARY_XPU_ID_ENV = "ROLL_PRIMARY_XPU_ID"


class XpuPlatform(Platform):
    device_name: str = "INTEL"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "ZE_AFFINITY_MASK"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR"
    communication_backend: str = "xccl"
    supports_tensor_ipc: bool = False

    @classmethod
    def is_xpu(cls) -> bool:
        return True

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        return

    @classmethod
    def set_allocator_settings(cls, env: str) -> None:
        return

    @classmethod
    def reset_max_memory_allocated(cls, device=None) -> None:
        torch.xpu.reset_peak_memory_stats(device)

    @classmethod
    def reset_max_memory_cached(cls, device=None) -> None:
        torch.xpu.reset_peak_memory_stats(device)

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        env_vars = {
            **Platform.get_common_envs(),
            "RAY_get_check_signal_interval_milliseconds": "1",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "RAY_CGRAPH_get_timeout": "600",
        }
        return env_vars

    @classmethod
    def update_env_vars_for_visible_devices(cls, env_vars: dict, gpu_ranks: list) -> None:
        env_vars.update(
            {
                ASSIGNED_XPU_IDS_ENV: ",".join(map(str, gpu_ranks)),
                PRIMARY_XPU_ID_ENV: str(gpu_ranks[0]),
                cls.ray_experimental_noset: "1",
            }
        )

    @classmethod
    def get_visible_gpus(cls) -> list:
        return os.environ.get(ASSIGNED_XPU_IDS_ENV, "").split(",")

    @classmethod
    def select_assigned_devices(cls) -> None:
        primary = os.environ.get(PRIMARY_XPU_ID_ENV)
        if primary is None:
            return
        if os.environ.get(cls.device_control_env_var):
            raise RuntimeError(
                f"{cls.device_control_env_var}={os.environ[cls.device_control_env_var]} narrows XPU visibility, "
                f"but {PRIMARY_XPU_ID_ENV}={primary} requires every node-local XPU to be visible"
            )
        device = int(primary)
        visible = torch.xpu.device_count()
        if device >= visible:
            raise RuntimeError(f"Ray assigned XPU {device}, but this process sees only {visible} XPUs")
        torch.xpu.set_device(device)
        current = torch.xpu.current_device()
        if current != device:
            raise RuntimeError(f"torch.xpu.set_device({device}) left the current XPU at {current}")
        logger.info(
            f"XPU placement: host={socket.gethostname()} pid={os.getpid()} rank={os.environ.get('RANK')} "
            f"assigned={os.environ.get(ASSIGNED_XPU_IDS_ENV)} current_device={current} "
            f"uuid={torch.xpu.get_device_properties(current).uuid}"
        )

    @classmethod
    def get_vllm_device_kwargs(cls) -> dict:
        assigned = os.environ.get(ASSIGNED_XPU_IDS_ENV)
        if not assigned:
            return {}
        import vllm
        from packaging.version import Version

        if Version(vllm.__version__) < Version("0.31.0"):
            raise RuntimeError(f"vLLM {vllm.__version__} ignores device_ids on XPU; vLLM >= 0.31 is required")
        return {"device_ids": [int(i) for i in assigned.split(",")]}

    @classmethod
    def get_vllm_worker_class(cls):
        try:
            from vllm.v1.worker.xpu_worker import XPUWorker

            logger.info("Successfully imported vLLM XPU Worker.")
            return XPUWorker
        except ImportError as e:
            logger.error("Failed to import vLLM Worker. Make sure vLLM is installed correctly: %s", e)
            raise RuntimeError("vLLM is not installed or not properly configured.") from e

    @classmethod
    def get_vllm_run_time_env_vars(cls, gpu_rank: str) -> dict:
        env_vars = {
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "ZE_AFFINITY_MASK": f"{gpu_rank}",
            "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR": "1",
        }
        return env_vars

    @classmethod
    def apply_ulysses_patch(cls) -> None:
        return
