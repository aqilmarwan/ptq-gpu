"""Variant loading, VRAM-budgeted caching, and the two generation backends.

`Registry` is what `main.py` talks to. It owns the variant catalog and a single
backend:

* `TensorRTBackend` -- real SDXL on CUDA via prebuilt TensorRT engines. There is
  no Hugging Face / diffusers dependency here: the text encoders, UNet, and VAE
  decoder run as `.plan` engines (fetched from S3 into the engine dir by the
  build pipeline / an init container), and the scheduler is vendored
  (`trt_runtime.EulerScheduler`). Keeps an LRU set of bundles hot in VRAM
  (``max_resident``) and measures latency / VRAM / throughput honestly.
* `DemoBackend` -- no engines, no GPU. Produces a seeded procedural image and
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

# SDXL constants (fixed by the base architecture).
VAE_SCALE = 0.13025
LATENT_CHANNELS = 4
VAE_DOWNSAMPLE = 8


@dataclass
class Serving:
    """How to serve a variant -- the `serving:` block from variants.yaml."""

    engine: str            # bundle key: <engine>/ under the engine dir (and S3)
    precision: str = "fp16"  # fp16 | int8 | fp8 -- informational; baked into the engine
    lora: Optional[str] = None  # baked into the engine at build time; label only


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
# TensorRT backend (real SDXL, prebuilt engines)
# --------------------------------------------------------------------------- #


@dataclass
class _Bundle:
    """One variant's loaded engines + tokenizers, kept hot in VRAM."""

    te1: object
    te2: object
    unet: object
    vae: object
    tokenizers: tuple


class TensorRTBackend:
    """Real inference from prebuilt TRT engines. HF-free, honest metrics.

    Expected per-variant bundle at ``config.ENGINE_DIR/<serving.engine>/``:
      text_encoder.plan  text_encoder_2.plan  unet.plan  vae_decoder.plan
      tokenizer/  tokenizer_2/  metadata.json

    Engine IO contract (produced by pipelines/build_engines.py):
      text_encoder(.2)   in: input_ids[B,77]      out: hidden_states, (te2) pooled
      unet               in: sample, timestep, encoder_hidden_states, text_embeds, time_ids
                         out: noise_pred    (batch 2, for classifier-free guidance)
      vae_decoder        in: latent         out: images[B,3,H,W] in [-1,1]
    """

    plane = "tensorrt"

    def __init__(self, engine_dir: Path, max_resident: int) -> None:
        self.engine_dir = engine_dir
        self.max_resident = max(1, max_resident)
        self._cache: "OrderedDict[str, _Bundle]" = OrderedDict()

    def is_resident(self, variant_id: str) -> bool:
        return variant_id in self._cache

    def _load_bundle(self, serving: Serving) -> _Bundle:
        from trt_runtime import ENGINE_FILES, TRTEngine, load_tokenizers

        bundle_dir = self.engine_dir / serving.engine
        if not bundle_dir.is_dir():
            raise FileNotFoundError(
                f"engine bundle not found: {bundle_dir}. It should be fetched from "
                f"{config.ENGINE_S3_URI or '<ENGINE_S3_URI unset>'} by the init container."
            )
        engines = {name: TRTEngine(bundle_dir / f"{name}.plan") for name in ENGINE_FILES}
        return _Bundle(
            te1=engines["text_encoder"],
            te2=engines["text_encoder_2"],
            unet=engines["unet"],
            vae=engines["vae_decoder"],
            tokenizers=load_tokenizers(bundle_dir),
        )

    def _ensure_loaded(self, variant: Variant, serving: Serving):
        """Return (bundle, was_cold, cold_load_ms)."""
        if variant.id in self._cache:
            self._cache.move_to_end(variant.id)
            return self._cache[variant.id], False, 0.0

        while len(self._cache) >= self.max_resident:
            evicted_id, _ = self._cache.popitem(last=False)
            log.info("Evicting %s to free VRAM", evicted_id)
            self._free()

        t0 = time.perf_counter()
        bundle = self._load_bundle(serving)
        cuda_sync()
        cold_ms = (time.perf_counter() - t0) * 1000.0
        self._cache[variant.id] = bundle
        return bundle, True, cold_ms

    @staticmethod
    def _free() -> None:
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(self, variant: Variant, serving: Serving, params: GenerationParams, emit: Emit) -> GenOutput:
        import torch

        from trt_runtime import EulerScheduler

        cold_before = not self.is_resident(variant.id)
        emit({
            "type": "status",
            "stage": "load",
            "message": (f"Cold-loading {variant.label} engines into VRAM..." if cold_before
                        else f"{variant.label} resident -- warm start"),
            "cold": cold_before,
        })
        bundle, cold, cold_ms = self._ensure_loaded(variant, serving)

        reset_peak_vram()
        gen = torch.Generator(device="cuda").manual_seed(int(params.seed))

        emit({"type": "status", "stage": "denoise", "message": "Running denoising loop...", "cold": cold})

        # --- text conditioning (uncond + cond, batched for guidance) --------
        prompt_embeds, pooled = self._encode(bundle, params.prompt, params.negative_prompt)
        add_time_ids = torch.tensor(
            [[params.height, params.width, 0, 0, params.height, params.width]] * 2,
            device="cuda", dtype=torch.float16,
        )

        # --- latents --------------------------------------------------------
        h = params.height // VAE_DOWNSAMPLE
        w = params.width // VAE_DOWNSAMPLE
        latents = torch.randn((1, LATENT_CHANNELS, h, w), generator=gen, device="cuda", dtype=torch.float16)
        sched = EulerScheduler()
        timesteps = sched.set_timesteps(params.steps)
        latents = latents * sched.init_noise_sigma

        # --- denoise loop ---------------------------------------------------
        cuda_sync()
        denoise_start = time.perf_counter()
        for i, t in enumerate(timesteps):
            model_in = sched.scale_model_input(torch.cat([latents, latents]), i)
            timestep = t.expand(2).to(torch.float16)
            noise = bundle.unet.infer({
                "sample": model_in,
                "timestep": timestep,
                "encoder_hidden_states": prompt_embeds,
                "text_embeds": pooled,
                "time_ids": add_time_ids,
            })["noise_pred"]
            noise_uncond, noise_cond = noise.chunk(2)
            noise = noise_uncond + params.guidance * (noise_cond - noise_uncond)
            latents = sched.step(noise, i, latents)
            emit({"type": "progress", "step": i + 1, "totalSteps": params.steps})
        cuda_sync()
        denoise_ms = (time.perf_counter() - denoise_start) * 1000.0

        # --- VAE decode -----------------------------------------------------
        emit({"type": "status", "stage": "decode", "message": "VAE decoding latents...", "cold": cold})
        vae_start = time.perf_counter()
        image_t = bundle.vae.infer({"latent": latents / VAE_SCALE})["images"]
        cuda_sync()
        vae_ms = (time.perf_counter() - vae_start) * 1000.0

        image = _tensor_to_image(image_t)
        throughput = params.steps / (denoise_ms / 1000.0) if denoise_ms > 0 else 0.0

        return GenOutput(
            image_url=_img_to_data_url(image),
            cold=cold,
            cold_load_ms=round(cold_ms),
            denoise_ms=round(denoise_ms),
            vae_ms=round(vae_ms),
            total_ms=round(cold_ms + denoise_ms + vae_ms),
            vram_peak_gb=peak_vram_gb(),
            throughput=round(throughput, 2),
        )

    def _encode(self, bundle: _Bundle, prompt: str, negative: str):
        """SDXL dual-encoder conditioning. Returns (prompt_embeds[2,77,2048], pooled[2,1280])."""
        import torch

        tok1, tok2 = bundle.tokenizers
        seqs, pooleds = [], []
        for text in (negative or "", prompt):  # uncond first, then cond
            hidden = []
            pooled_p = None
            for tok, engine, is_two in ((tok1, bundle.te1, False), (tok2, bundle.te2, True)):
                ids = tok(text, padding="max_length", max_length=77, truncation=True,
                          return_tensors="pt").input_ids.to("cuda").to(torch.int32)
                out = engine.infer({"input_ids": ids})
                hidden.append(out["hidden_states"])
                if is_two:
                    pooled_p = out["pooled"]
            seqs.append(torch.cat(hidden, dim=-1))
            pooleds.append(pooled_p)
        return torch.cat(seqs, dim=0), torch.cat(pooleds, dim=0)


def _tensor_to_image(image_t):
    """(1,3,H,W) in [-1,1] -> PIL RGB."""
    import numpy as np
    from PIL import Image

    arr = (image_t / 2 + 0.5).clamp(0, 1)[0]
    arr = (arr.permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr, "RGB")


def _trt_available() -> bool:
    try:
        import tensorrt  # noqa: F401

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class Registry:
    def __init__(self, variants_file: Path | None = None) -> None:
        path = Path(variants_file or config.VARIANTS_FILE)
        doc = yaml.safe_load(path.read_text())

        max_resident = int(doc.get("max_resident", 2))
        # `base_model` is the HF id used to build engines; the UI shows base_label.
        base = doc.get("base_label", "SDXL 1.0")

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
                engine=sv.get("engine", raw["id"]),
                precision=sv.get("precision", "fp16"),
                lora=sv.get("lora"),
            )
        self._by_id = {v.id: v for v in self._variants}

        use_real = not config.FORCE_DEMO and cuda_available() and _trt_available()
        if use_real:
            log.info("CUDA + TensorRT detected -- using TensorRTBackend (real metrics)")
            self.backend = TensorRTBackend(config.ENGINE_DIR, max_resident)
        else:
            if config.FORCE_DEMO:
                reason = "STUDIO_DEMO set"
            elif not cuda_available():
                reason = "no CUDA device"
            else:
                reason = "tensorrt not importable"
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
