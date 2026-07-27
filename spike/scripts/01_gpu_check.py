"""M0 deliverable 1 — GPU environment verification.

Validates RSK-7: the RTX 5060 is a Blackwell-generation GPU (compute capability
sm_120). PyTorch wheels built before CUDA 12.8 do not contain sm_120 kernels and
will either fail outright or fall back to slow PTX JIT compilation.

Checking `torch.cuda.is_available()` alone is NOT sufficient — it can return True
while the installed build has no native kernels for this architecture. The
authoritative check is whether sm_120 appears in `torch.cuda.get_arch_list()`.

Run:  uv run python scripts/01_gpu_check.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results" / "01_gpu.json"


def _fmt_gb(num_bytes: int) -> float:
    return round(num_bytes / (1024**3), 2)


def main() -> int:
    out: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed.")
        return 1

    out["torch_version"] = torch.__version__
    out["torch_cuda_build"] = torch.version.cuda
    print(f"torch                {torch.__version__}")
    print(f"built against CUDA   {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("\nFAIL: torch.cuda.is_available() is False.")
        print("  Likely a CPU-only wheel. Reinstall from the CUDA 12.8+ index:")
        print("  uv pip install torch --index-url https://download.pytorch.org/whl/cu128")
        out["cuda_available"] = False
        RESULTS.write_text(json.dumps(out, indent=2))
        return 1

    out["cuda_available"] = True

    # --- Device identity -----------------------------------------------------
    props = torch.cuda.get_device_properties(0)
    cc = f"{props.major}.{props.minor}"
    sm = f"sm_{props.major}{props.minor}"
    total_gb = _fmt_gb(props.total_memory)

    out |= {
        "device_name": props.name,
        "compute_capability": cc,
        "sm": sm,
        "vram_total_gb": total_gb,
        "multiprocessors": props.multi_processor_count,
    }
    print(f"device               {props.name}")
    print(f"compute capability   {cc}  ({sm})")
    print(f"VRAM total           {total_gb} GB")

    # --- The Blackwell check (RSK-7) ----------------------------------------
    arch_list = torch.cuda.get_arch_list()
    out["arch_list"] = arch_list
    native = sm in arch_list
    out["native_kernels"] = native

    print(f"\nbuilt-in archs       {', '.join(arch_list)}")
    if native:
        print(f"OK: native {sm} kernels present.")
    else:
        print(f"WARN: {sm} NOT in the build's arch list.")
        print("  Ops may run via PTX JIT (slow first call) or fail outright.")
        print("  Benchmark results below are then NOT representative.")

    # --- Real tensor op on device -------------------------------------------
    # is_available() lying is exactly the failure mode we are guarding against,
    # so actually compute something and verify the numbers.
    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
        c = a @ b
        torch.cuda.synchronize()
        ref = (a.cpu() @ b.cpu())
        max_err = (c.cpu() - ref).abs().max().item()
        out["matmul_ok"] = True
        out["matmul_max_abs_err"] = max_err
        print(f"\nmatmul 2048x2048     OK (max abs err vs CPU: {max_err:.2e})")
    except Exception as exc:  # noqa: BLE001 - spike diagnostic
        out["matmul_ok"] = False
        out["matmul_error"] = repr(exc)
        print(f"\nFAIL: device matmul raised {exc!r}")
        RESULTS.write_text(json.dumps(out, indent=2))
        return 1

    # --- Throughput sanity check (fp16, closer to inference workload) --------
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    y = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(3):  # warm up / trigger any JIT
        _ = x @ y
    torch.cuda.synchronize()

    iters = 20
    start = time.perf_counter()
    for _ in range(iters):
        _ = x @ y
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # 2*M*N*K FLOPs per matmul
    tflops = (2 * 4096**3 * iters) / elapsed / 1e12
    out["fp16_matmul_tflops"] = round(tflops, 1)
    print(f"fp16 4096^3 matmul   {tflops:.1f} TFLOP/s")

    # --- Memory headroom -----------------------------------------------------
    free_b, total_b = torch.cuda.mem_get_info()
    out["vram_free_gb"] = _fmt_gb(free_b)
    print(f"VRAM free            {_fmt_gb(free_b)} GB of {_fmt_gb(total_b)} GB")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}")

    return 0 if native else 2


if __name__ == "__main__":
    raise SystemExit(main())
