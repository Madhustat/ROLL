"""Per-XPU health check: matmul, pageable round trip, pinned round trip. No collectives, no process group.
Run before the reproducer and again after any killed hang."""
import torch

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
    print(f"xpu:{d} {torch.xpu.get_device_name(d)}: matmul={mm} pageable_roundtrip={pageable} pinned_roundtrip={pinned}")
print("HEALTH OK" if ok else "HEALTH FAILED")
raise SystemExit(0 if ok else 1)
