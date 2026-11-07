"""XPU detection + version report for the ROLL env. Device work = the health patterns proven safe here
(matmul, blocking pageable round trip, pinned round trip); no collectives, no process group."""
import ctypes
import importlib.metadata as md
import os
import sys

import torch

print("=== package versions (pip metadata)")
for p in ["torch", "torchvision", "torchaudio", "triton", "triton-xpu", "vllm", "vllm-xpu-kernels", "oneccl", "oneccl-devel",
          "intel-sycl-rt", "intel-cmplr-lib-ur", "intel-cmplr-lib-rt", "dpcpp-cpp-rt", "intel-opencl-rt", "intel-openmp",
          "intel-pti", "transformers", "ray", "numpy"]:
    try:
        print(f"  {p:20s} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"  {p:20s} NOT INSTALLED")

print("=== torch")
print("  torch.__version__     ", torch.__version__)
print("  torch.version.xpu     ", torch.version.xpu)
print("  torch.version.cuda    ", torch.version.cuda)
print("  xpu.is_available()    ", torch.xpu.is_available())
print("  xpu.device_count()    ", torch.xpu.device_count())
print("  is_xccl_available()   ", torch.distributed.is_xccl_available())
for d in range(torch.xpu.device_count()):
    p = torch.xpu.get_device_properties(d)
    print(f"  xpu:{d} {p.name} | total_memory {p.total_memory / 2**30:.2f} GiB | driver {p.driver_version} | "
          f"platform {p.platform_name} | compute units {p.max_compute_units}")

print("=== oneCCL")
venv_lib = os.path.join(sys.prefix, "lib")
lib = ctypes.CDLL(os.path.join(venv_lib, "libccl.so.1"))


class LibraryVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint), ("minor", ctypes.c_uint), ("update", ctypes.c_uint),
                ("product_status", ctypes.c_char_p), ("build_date", ctypes.c_char_p), ("full", ctypes.c_char_p),
                ("rest", ctypes.c_ubyte * 256)]


fn = getattr(lib, "_ZN3ccl5utils19get_library_versionEv")
fn.restype = LibraryVersion
v = fn()
print(f"  libccl.so.1 reports   {v.major}.{v.minor}.{v.update} {v.product_status.decode()}")
print(f"  full                  {v.full.decode()}")

print("=== loaded native libraries (this process)")
maps = {line.split()[-1] for line in open("/proc/self/maps") if ".so" in line}
for k in ("libccl.so", "libsycl.so", "libur_loader.so", "libze_loader.so", "libze_intel_gpu.so", "libtorch_xpu.so"):
    for path in sorted(m for m in maps if os.path.basename(m).startswith(k)):
        print(f"  {path}")

print("=== vLLM")
import vllm  # noqa: E402
from vllm.platforms import current_platform as vp  # noqa: E402

print("  vllm.__version__      ", vllm.__version__)
print("  vllm platform         ", type(vp).__name__, "| device_type", vp.device_type, "| is_xpu", vp.is_xpu())
for d in range(torch.xpu.device_count()):
    print(f"  vllm sees xpu:{d}      {vp.get_device_name(d)} | {vp.get_device_total_memory(d) / 2**30:.2f} GiB")

print("=== ROLL")
from roll.platforms import current_platform as rp  # noqa: E402

print("  roll current_platform ", type(rp).__name__, "| device_type", rp.device_type)

print("=== env")
for k in ("ONEAPI_DEVICE_SELECTOR", "ZE_AFFINITY_MASK", "CCL_ROOT", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
          "CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD", "CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD", "LD_LIBRARY_PATH"):
    print(f"  {k:38s} {os.environ.get(k)}")

print("=== per-XPU health (matmul, pageable round trip, pinned round trip)")
ok = True
for d in range(torch.xpu.device_count()):
    dev = f"xpu:{d}"
    a, b = torch.randn(2048, 2048, device=dev), torch.randn(2048, 2048, device=dev)
    mm = torch.allclose((a @ b).cpu(), a.cpu() @ b.cpu(), atol=1e-2, rtol=1e-3)
    x = torch.arange(1 << 24, dtype=torch.float32)
    pageable = torch.equal(x.to(dev).cpu(), x)
    pin = x.pin_memory(device=dev)
    y = torch.empty_like(pin).pin_memory(device=dev)
    y.copy_(pin.to(dev, non_blocking=True), non_blocking=True)
    torch.xpu.synchronize(dev)
    pinned = torch.equal(y, x)
    ok &= mm and pageable and pinned
    print(f"  xpu:{d}: matmul={mm} pageable_roundtrip={pageable} pinned_roundtrip={pinned}")
print("HEALTH OK" if ok else "HEALTH FAILED")
raise SystemExit(0 if ok else 1)
