"""GPU-only execution: every model runs on CUDA; no silent CPU fallback."""
from __future__ import annotations

DEVICE = "cuda"


class GPUUnavailable(RuntimeError):
    pass


def require_gpu() -> dict:
    """Fail fast if CUDA is missing, enable TF32 math, and describe the GPU."""
    import torch

    if not torch.cuda.is_available():
        raise GPUUnavailable(
            "No CUDA GPU detected. This pipeline runs on the GPU only: install the NVIDIA driver "
            "and the CUDA build of PyTorch (see setup_env.ps1)."
        )
    torch.backends.cuda.matmul.allow_tf32 = True   # Ampere+: faster matmuls at negligible precision cost
    torch.backends.cudnn.allow_tf32 = True
    free, total = torch.cuda.mem_get_info()
    return {
        "name": torch.cuda.get_device_name(0),
        "vram_gb": round(total / 1024 ** 3, 1),
        "free_gb": round(free / 1024 ** 3, 1),
        "cuda": torch.version.cuda,
        "torch": torch.__version__.split("+")[0],
    }
