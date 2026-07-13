"""Offline: SDXL (+ optional LoRA) -> ONNX -> TensorRT engine bundle -> S3.

This is the *build* side of the TensorRT migration. HF is used here, once, to
pull the base weights and export them to ONNX; the resulting `.plan` engines are
what the serving plane runs (no HF/diffusers at request time). Invoked per
variant by pipelines/build_flow.py's build step on the real plane.

A bundle directory (published to ``s3://.../<engine>/``) contains:
    text_encoder.plan  text_encoder_2.plan  unet.plan  vae_decoder.plan
    tokenizer/  tokenizer_2/  metadata.json

Precision -> TensorRT builder flags:
    fp16 -> FP16
    int8 -> INT8 (needs a calibration cache; entropy calibration offline)
    fp8  -> FP8  (Ada/Hopper only; falls back to FP16 with a warning elsewhere)

GPU-only and heavy. Everything is imported lazily. The engine-build specifics
(opset, dynamic axes, calibration) follow NVIDIA's SDXL demo conventions but must
be validated on-device before production.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("studio.build_engines")

# Static input shapes for a 1024x1024 SDXL at CFG batch 2 (uncond+cond).
_B, _SEQ, _LAT, _H, _W = 2, 77, 128, 1024, 1024


@dataclass
class EngineSpec:
    engine: str            # bundle key (== serving.engine)
    precision: str         # fp16 | int8 | fp8
    lora: Optional[str] = None  # LoRA name to fuse before export


def build_bundle(spec: EngineSpec, base_model: str, out_root: Path, lora_dir: Path) -> Path:
    """Build a full engine bundle for one variant. Returns its local directory."""
    import torch
    from diffusers import StableDiffusionXLPipeline

    bundle = out_root / spec.engine
    bundle.mkdir(parents=True, exist_ok=True)
    log.info("Building %s bundle (%s%s)", spec.engine, spec.precision,
             f", lora={spec.lora}" if spec.lora else "")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    if spec.lora:
        lora_path = lora_dir / f"{spec.lora}.safetensors"
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA {lora_path} missing -- train it first (build_flow train step).")
        pipe.load_lora_weights(str(lora_path))
        pipe.fuse_lora()  # bake weights in so the exported engine carries the style

    onnx_dir = bundle / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    components = {
        "text_encoder": _export_text_encoder(pipe.text_encoder, onnx_dir, "text_encoder", pooled=False),
        "text_encoder_2": _export_text_encoder(pipe.text_encoder_2, onnx_dir, "text_encoder_2", pooled=True),
        "unet": _export_unet(pipe.unet, onnx_dir),
        "vae_decoder": _export_vae_decoder(pipe.vae, onnx_dir),
    }
    for name, onnx_path in components.items():
        _build_engine(onnx_path, bundle / f"{name}.plan", spec.precision)

    # Tokenizers travel with the bundle so serving loads them offline.
    pipe.tokenizer.save_pretrained(str(bundle / "tokenizer"))
    pipe.tokenizer_2.save_pretrained(str(bundle / "tokenizer_2"))
    (bundle / "metadata.json").write_text(json.dumps({
        "engine": spec.engine, "precision": spec.precision, "lora": spec.lora,
        "base_model": base_model, "resolution": _H,
    }, indent=2))

    del pipe
    torch.cuda.empty_cache()
    return bundle


# --------------------------------------------------------------------------- #
# ONNX export (one function per component; dynamic batch on the token dim)
# --------------------------------------------------------------------------- #


def _export_text_encoder(model, onnx_dir: Path, name: str, pooled: bool) -> Path:
    import torch

    path = onnx_dir / f"{name}.onnx"
    model = model.eval()
    ids = torch.zeros((1, _SEQ), dtype=torch.int32, device=model.device)
    outputs = ["hidden_states"] + (["pooled"] if pooled else [])

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, input_ids):
            out = self.m(input_ids, output_hidden_states=True)
            hs = out.hidden_states[-2]                 # SDXL uses the penultimate layer
            return (hs, out[0]) if pooled else (hs,)

    torch.onnx.export(
        _Wrap(model), (ids,), str(path),
        input_names=["input_ids"], output_names=outputs,
        dynamic_axes={"input_ids": {0: "B"}, **{o: {0: "B"} for o in outputs}},
        opset_version=18,
    )
    return path


def _export_unet(unet, onnx_dir: Path) -> Path:
    import torch

    path = onnx_dir / "unet.onnx"
    unet = unet.eval()
    dev, dt = unet.device, torch.float16
    dummies = (
        torch.randn(_B, 4, _LAT, _LAT, dtype=dt, device=dev),          # sample
        torch.tensor([1.0, 1.0], dtype=dt, device=dev),                # timestep
        torch.randn(_B, _SEQ, 2048, dtype=dt, device=dev),             # encoder_hidden_states
    )
    added = {"text_embeds": torch.randn(_B, 1280, dtype=dt, device=dev),
             "time_ids": torch.randn(_B, 6, dtype=dt, device=dev)}

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
            return self.m(sample, timestep, encoder_hidden_states,
                          added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids}).sample

    torch.onnx.export(
        _Wrap(unet), (*dummies, added["text_embeds"], added["time_ids"]), str(path),
        input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
        output_names=["noise_pred"],
        dynamic_axes={k: {0: "B"} for k in
                      ["sample", "encoder_hidden_states", "text_embeds", "time_ids", "noise_pred"]},
        opset_version=18,
    )
    return path


def _export_vae_decoder(vae, onnx_dir: Path) -> Path:
    import torch

    path = onnx_dir / "vae_decoder.onnx"
    vae = vae.eval()
    latent = torch.randn(1, 4, _LAT, _LAT, dtype=torch.float16, device=vae.device)

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, latent):
            return self.m.decode(latent).sample

    torch.onnx.export(
        _Wrap(vae), (latent,), str(path),
        input_names=["latent"], output_names=["images"],
        dynamic_axes={"latent": {0: "B"}, "images": {0: "B"}},
        opset_version=18,
    )
    return path


# --------------------------------------------------------------------------- #
# TensorRT engine build
# --------------------------------------------------------------------------- #


def _build_engine(onnx_path: Path, plan_path: Path, precision: str) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TRT 10 networks are always explicit-batch (the flag was removed); TRT 8
    # needs it set. Handle both so the same build box works across versions.
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path.name}: {errs}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8 GiB
    if precision in {"fp16", "int8", "fp8"}:
        cfg.set_flag(trt.BuilderFlag.FP16)          # fp16 fallback for unsupported layers
    if precision == "int8":
        cfg.set_flag(trt.BuilderFlag.INT8)          # requires a calibrator; see note below
    if precision == "fp8":
        cfg.set_flag(trt.BuilderFlag.FP8)           # Ada/Hopper

    # NOTE: INT8 needs an IInt8EntropyCalibrator2 over representative latents to
    # populate scales; FP8 typically pairs with per-tensor scales baked at export
    # (ModelOpt). Wire your calibrator here before shipping INT8/FP8 engines.

    _add_dynamic_profile(builder, network, cfg, trt)

    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        raise RuntimeError(f"engine build returned None for {onnx_path.name}")
    plan_path.write_bytes(serialized)
    log.info("  built %s (%.0f MB, %s)", plan_path.name, len(serialized) / 1e6, precision)


def _add_dynamic_profile(builder, network, cfg, trt) -> None:
    """Fix the batch dim to 2 (CFG); text encoders run batch 1..2."""
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = list(inp.shape)
        lo = [1 if d == -1 else d for d in shape]
        opt = [2 if d == -1 else d for d in shape]
        hi = [2 if d == -1 else d for d in shape]
        profile.set_shape(inp.name, lo, opt, hi)
    cfg.add_optimization_profile(profile)


# --------------------------------------------------------------------------- #
# Publish
# --------------------------------------------------------------------------- #


def publish_s3(bundle: Path, s3_uri: str) -> None:
    """Upload a bundle to ``<s3_uri>/<bundle-name>/`` (skips the local onnx/ dir)."""
    import boto3

    assert s3_uri.startswith("s3://"), s3_uri
    bucket, _, prefix = s3_uri[len("s3://"):].partition("/")
    key_root = f"{prefix.rstrip('/')}/{bundle.name}" if prefix else bundle.name
    s3 = boto3.client("s3")
    for path in bundle.rglob("*"):
        if path.is_dir() or "onnx" in path.relative_to(bundle).parts:
            continue
        key = f"{key_root}/{path.relative_to(bundle).as_posix()}"
        s3.upload_file(str(path), bucket, key)
    log.info("  published %s -> s3://%s/%s", bundle.name, bucket, key_root)
