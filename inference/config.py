"""Runtime configuration, sourced from the environment.

The service has two execution planes:

* **real** — diffusers + torch on a CUDA GPU, measuring honest metrics.
* **demo** — a CPU-friendly fallback (no model download) that simulates the
  pipeline so the frontend is fully exercisable in local dev. Demo metrics are
  derived from the registry benchmark and are clearly logged as simulated.

The real plane is selected automatically when CUDA is available and
``STUDIO_DEMO`` is not set.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


# Force the demo plane even if a GPU is present (useful for UI work).
FORCE_DEMO: bool = _flag("STUDIO_DEMO")

# Path to the variant catalog.
VARIANTS_FILE: Path = Path(os.getenv("STUDIO_VARIANTS", str(HERE / "variants.yaml")))

# Where LoRA weights referenced by variants.yaml live.
LORA_DIR: Path = Path(os.getenv("STUDIO_LORA_DIR", str(HERE / "loras")))

# Browser origins allowed to call the API (the Next.js dev + preview servers).
CORS_ORIGINS: list[str] = _list(
    "STUDIO_CORS_ORIGINS", "http://localhost:3000,http://localhost:3100"
)

# Generation guard-rails (also clamped per-request).
MAX_STEPS: int = int(os.getenv("STUDIO_MAX_STEPS", "60"))
MAX_DIM: int = int(os.getenv("STUDIO_MAX_DIM", "1024"))

# Hugging Face cache / base model override.
BASE_MODEL_ENV: str | None = os.getenv("STUDIO_BASE_MODEL") or None
