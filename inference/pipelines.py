"""Variant loading, VRAM-budgeted caching, and the two generation backends.

`Registry` is what `main.py` talks to. It owns the variant catalog and a single
backend:

* `DiffusersBackend` -- real SDXL on CUDA. Lazily loads each variant, keeps an
  LRU set hot in VRAM (``max_resident``), quantises per `variants.yaml`, and
  measures latency / VRAM / throughput around the real denoise loop.
* `DemoBackend` -- no model, no GPU. Produces a seeded procedural image and
  simulates timing from the registry benchmark. Used automatically off-GPU so
  the frontend is fully exercisable in local dev. Its metrics are simulated and
  logged as such.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml

import config
from metrics import cuda_available, cuda_sync, peak_vram_gb, reset_peak_vram
from schemas import GenerationParams, GenerationResult, LatencyBreakdown, Metrics, Variant

log = logging.getLogger("studio.pipelines")

# An emit callback receives event dicts (status / progress) to stream as SSE.
Emit = Callable[[dict], None]


@dataclass
class Serving:
    """How to load a variant -- the `serving:` block from variants.yaml."""

    quant: str = "none"  # none | int8 | nf4
    lora: Optional[str] = None
    lora_scale: float = 0.8


@dataclass
class GenOutput:
    image_url: str
    cold: bool
    cold_load_ms: int
    denoise_ms: int
    vae_ms: int
    total_ms: int
    vram_peak_gb: float
    throughput: float


def _img_to_data_url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# Demo backend
# --------------------------------------------------------------------------- #


class DemoBackend:
    """CPU fallback: seeded procedural art + simulated metrics."""

    plane = "demo"

    def __init__(self) -> None:
        self._resident: set[str] = set()

    def is_resident(self, variant_id: str) -> bool:
        return variant_id in self._resident

    def generate(self, variant: Variant, serving: Serving, params: GenerationParams, emit: Emit) -> GenOutput:
        cold = variant.id not in self._resident
        cold_load_ms = (1400 + variant.size_gb * 230) if cold else 0.0

        emit({
            "type": "status",
            "stage": "load",
            "message": (f"Cold-loading {variant.label} weights into VRAM..." if cold
                        else f"{variant.label} resident -- warm start"),
            "cold": cold,
        })
        time.sleep(0.5 if cold else 0.12)
        self._resident.add(variant.id)

        emit({"type": "status", "stage": "denoise", "message": "Running denoising loop...", "cold": cold})
        per_step = min(1000.0 / variant.steps_per_sec, 90.0) / 1000.0
        for step in range(1, params.steps + 1):
            time.sleep(per_step)
            emit({"type": "progress", "step": step, "totalSteps": params.steps})

        emit({"type": "status", "stage": "decode", "message": "VAE decoding latents...", "cold": cold})
        time.sleep(0.2)

        denoise_ms = (params.steps / variant.steps_per_sec) * 1000.0
        vae_ms = 180 + variant.vram_gb * 6
        total_ms = cold_load_ms + denoise_ms + vae_ms
        image = _demo_image(params.prompt, params.seed, variant.quality, params.width, params.height)

        return GenOutput(
            image_url=_img_to_data_url(image),
            cold=cold,
            cold_load_ms=round(cold_load_ms),
            denoise_ms=round(denoise_ms),
            vae_ms=round(vae_ms),
            total_ms=round(total_ms),
            vram_peak_gb=round(variant.vram_gb + 0.3, 1),
            throughput=variant.steps_per_sec,
        )


def _demo_image(prompt: str, seed: int, quality: int, width: int, height: int):
    """Deterministic abstract artwork. Lower quality -> visibly grainier, so the
    quant tradeoff is visible even without a GPU. Pure function of (prompt, seed)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng((abs(hash(prompt)) ^ (seed * 2654435761)) & 0xFFFFFFFF)
    palettes = [
        [(30, 27, 75), (124, 58, 237), (34, 211, 238), (240, 171, 252)],
        [(12, 74, 110), (14, 165, 233), (94, 234, 212), (253, 230, 138)],
        [(74, 4, 78), (219, 39, 119), (251, 113, 133), (253, 186, 116)],
        [(5, 46, 22), (22, 163, 74), (163, 230, 53), (253, 224, 71)],
        [(15, 23, 42), (99, 102, 241), (56, 189, 248), (224, 231, 255)],
    ]
    palette = palettes[int(rng.integers(0, len(palettes)))]

    w, h = max(64, width), max(64, height)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xs /= w
    ys /= h
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    base = np.array(palette[0], dtype=np.float32) / 255.0
    canvas += base * (1.0 - ys[..., None] * 0.8)

    for i in range(5):
        cx, cy = float(rng.random()), float(rng.random())
        r = 0.18 + float(rng.random()) * 0.30
        color = np.array(palette[(i + 1) % len(palette)], dtype=np.float32) / 255.0
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        falloff = np.exp(-d2 / (2 * r * r))[..., None]
        canvas += color * falloff * (0.5 + 0.45 * float(rng.random()))

    grain = (100 - quality) / 100.0 * 0.22
    if grain > 0:
        canvas += rng.normal(0, grain, canvas.shape).astype(np.float32)

    canvas = np.clip(canvas, 0, 1)
    return Image.fromarray((canvas * 255).astype("uint8"), "RGB")


# --------------------------------------------------------------------------- #
# Diffusers backend (real SDXL on CUDA)
# --------------------------------------------------------------------------- #


class DiffusersBackend:
    """Real inference. Lazy per-variant load, LRU VRAM cache, honest metrics."""

    plane = "cuda"

    def __init__(self, base_model: str, max_resident: int) -> None:
        self.base_model = base_model
        self.max_resident = max(1, max_resident)
        self._cache: "OrderedDict[str, object]" = OrderedDict()  # variant_id -> pipeline

    def is_resident(self, variant_id: str) -> bool:
        return variant_id in self._cache

    def _build_pipeline(self, variant: Variant, serving: Serving):
        import torch
        from diffusers import StableDiffusionXLPipeline

        log.info("Loading %s (%s) ...", variant.id, serving.quant)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        if serving.quant in {"int8", "nf4"}:
            # Weight-only quantisation of the UNet; VAE + text encoders stay FP16.
            from optimum.quanto import freeze, qint4, qint8, quantize

            weights = qint8 if serving.quant == "int8" else qint4
            quantize(pipe.unet, weights=weights)
            freeze(pipe.unet)

        pipe = pipe.to("cuda")

        if serving.lora:
            lora_path = config.LORA_DIR / Path(serving.lora).name
            if lora_path.exists():
                pipe.load_lora_weights(str(lora_path))
            else:
                log.warning("LoRA weights not found at %s -- serving base weights", lora_path)

        pipe.set_progress_bar_config(disable=True)
        return pipe

    def _ensure_loaded(self, variant: Variant, serving: Serving):
        """Return (pipeline, was_cold, cold_load_ms)."""
        if variant.id in self._cache:
            self._cache.move_to_end(variant.id)
            return self._cache[variant.id], False, 0.0

        while len(self._cache) >= self.max_resident:
            evicted_id, evicted = self._cache.popitem(last=False)
            log.info("Evicting %s to free VRAM", evicted_id)
            self._free(evicted)

        t0 = time.perf_counter()
        pipe = self._build_pipeline(variant, serving)
        cuda_sync()
        cold_ms = (time.perf_counter() - t0) * 1000.0
        self._cache[variant.id] = pipe
        return pipe, True, cold_ms

    @staticmethod
    def _free(pipe) -> None:
        try:
            import torch

            del pipe
            torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(self, variant: Variant, serving: Serving, params: GenerationParams, emit: Emit) -> GenOutput:
        import torch

        cold_before = not self.is_resident(variant.id)
        emit({
            "type": "status",
            "stage": "load",
            "message": (f"Cold-loading {variant.label} into VRAM..." if cold_before
                        else f"{variant.label} resident -- warm start"),
            "cold": cold_before,
        })
        pipe, cold, cold_ms = self._ensure_loaded(variant, serving)

        reset_peak_vram()
        generator = torch.Generator(device="cuda").manual_seed(int(params.seed))

        emit({"type": "status", "stage": "denoise", "message": "Running denoising loop...", "cold": cold})
        timings: dict[str, float] = {}

        def on_step_end(pipe_, step_index, timestep, callback_kwargs):
            if step_index == 0:
                timings["denoise_start"] = time.perf_counter()
            emit({"type": "progress", "step": step_index + 1, "totalSteps": params.steps})
            return callback_kwargs

        call_start = time.perf_counter()
        cross_attn = {"scale": serving.lora_scale} if serving.lora else None
        out = pipe(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt or None,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            width=params.width,
            height=params.height,
            generator=generator,
            cross_attention_kwargs=cross_attn,
            callback_on_step_end=on_step_end,
        )
        cuda_sync()
        denoise_end = time.perf_counter()

        emit({"type": "status", "stage": "decode", "message": "VAE decoding latents...", "cold": cold})

        denoise_start = timings.get("denoise_start", call_start)
        denoise_ms = (denoise_end - denoise_start) * 1000.0
        vae_ms = max(0.0, (denoise_end - call_start) * 1000.0 - denoise_ms)
        total_ms = cold_ms + (denoise_end - call_start) * 1000.0
        throughput = params.steps / (denoise_ms / 1000.0) if denoise_ms > 0 else 0.0

        image = out.images[0]
        return GenOutput(
            image_url=_img_to_data_url(image),
            cold=cold,
            cold_load_ms=round(cold_ms),
            denoise_ms=round(denoise_ms),
            vae_ms=round(vae_ms),
            total_ms=round(total_ms),
            vram_peak_gb=peak_vram_gb(),
            throughput=round(throughput, 2),
        )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class Registry:
    def __init__(self, variants_file: Path | None = None) -> None:
        path = Path(variants_file or config.VARIANTS_FILE)
        doc = yaml.safe_load(path.read_text())

        base = config.BASE_MODEL_ENV or doc["base_model"]
        max_resident = int(doc.get("max_resident", 2))

        self._variants: list[Variant] = []
        self._serving: dict[str, Serving] = {}
        for raw in doc["variants"]:
            reg = raw["registry"]
            self._variants.append(Variant(
                id=raw["id"],
                label=raw["label"],
                precision=raw["precision"],
                style=raw["style"],
                base=base,
                lora_name=raw.get("loraName"),
                size_gb=reg["sizeGB"],
                vram_gb=reg["vramGB"],
                steps_per_sec=reg["stepsPerSec"],
                quality=reg["quality"],
                default_steps=reg["defaultSteps"],
                blurb=raw["blurb"],
                licence=raw["licence"],
            ))
            sv = raw.get("serving", {})
            self._serving[raw["id"]] = Serving(
                quant=sv.get("quant", "none"),
                lora=sv.get("lora"),
                lora_scale=sv.get("lora_scale", 0.8),
            )
        self._by_id = {v.id: v for v in self._variants}

        use_real = not config.FORCE_DEMO and cuda_available()
        if use_real:
            log.info("CUDA detected -- using DiffusersBackend (real metrics)")
            self.backend = DiffusersBackend(base, max_resident)
        else:
            reason = "STUDIO_DEMO set" if config.FORCE_DEMO else "no CUDA device"
            log.warning("DEMO plane (%s) -- metrics are simulated, not measured", reason)
            self.backend = DemoBackend()

    @property
    def plane(self) -> str:
        return self.backend.plane

    def variants(self) -> list[Variant]:
        return self._variants

    def has(self, variant_id: str) -> bool:
        return variant_id in self._by_id

    def run(self, params: GenerationParams, emit: Emit) -> GenerationResult:
        variant = self._by_id[params.variant_id]
        serving = self._serving[variant.id]
        out = self.backend.generate(variant, serving, params, emit)
        
        buf = io.BytesIO()
        out.save(buf, format='PNG')    
        return GenerationResult(
            image_url=out.image_url,
            variant_id=variant.id,
            params=params,
            metrics=Metrics(
                cold=out.cold,
                latency_ms=LatencyBreakdown(
                    cold_load=out.cold_load_ms,
                    denoise=out.denoise_ms,
                    vae_decode=out.vae_ms,
                    total=out.total_ms,
                ),
                throughput_steps_per_sec=out.throughput,
                vram_peak_gb=out.vram_peak_gb,
            ),
        )
