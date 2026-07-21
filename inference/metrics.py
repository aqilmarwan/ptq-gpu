"""Server-side measurement helpers.

Everything the result card shows is measured here, around the *real* work -- never
estimated. Cold vs warm is reported honestly by the pipeline registry.
"""

from __future__ import annotations


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def reset_peak_vram() -> None:
    """Zero the CUDA peak-allocation counter before a generation."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except Exception:
        pass


def peak_vram_gb() -> float:
    """Peak VRAM allocated since the last reset, in GB (0.0 off-GPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return round(torch.cuda.max_memory_allocated() / (1024**3), 2)
    except Exception:
        pass
    return 0.0


def cuda_sync() -> None:
    """Block until queued CUDA work finishes -- required for honest timing."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
