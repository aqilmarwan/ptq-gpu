"""Server-side measurement helpers.

Everything the result card shows is measured here, around the *real* work -- never
estimated. Cold vs warm is reported honestly by the pipeline registry.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


def now() -> float:
    """Monotonic clock in seconds -- immune to wall-clock adjustments."""
    return time.perf_counter()


@dataclass
class Stopwatch:
    """Accumulating timer. ``ms`` returns elapsed milliseconds."""

    _start: float = field(default_factory=now)
    _elapsed: float = 0.0
    _running: bool = True

    def stop(self) -> float:
        if self._running:
            self._elapsed += now() - self._start
            self._running = False
        return self.ms

    def lap(self) -> float:
        """Milliseconds since the last lap/start, without stopping."""
        t = now()
        delta = t - self._start
        self._start = t
        return delta * 1000.0

    @property
    def ms(self) -> float:
        live = (now() - self._start) if self._running else 0.0
        return (self._elapsed + live) * 1000.0


@contextmanager
def measure():
    """`with measure() as sw: ...` -> sw.ms holds the elapsed milliseconds."""
    sw = Stopwatch()
    try:
        yield sw
    finally:
        sw.stop()


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
