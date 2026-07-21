"""DreamBooth-LoRA training for SDXL, invoked by the build flow's `train` step.

Trains low-rank adapters on the SDXL UNet's attention layers against a small set
of instance images and exports a `.safetensors` file in the standard diffusers
layout -- i.e. exactly what `inference/pipelines.py` loads via
``pipe.load_lora_weights(...)`` and what `variants.yaml` points `serving.lora` at.

Scope: UNet-only LoRA (text encoders frozen), single instance concept, constant
LR. That's the honest 80/20 of DreamBooth-LoRA; it deliberately omits prior
preservation and text-encoder training. Runs on CUDA only -- there is no demo
path here, because a simulated/empty adapter would be a fake weight. The build
flow simply skips this step off-GPU.

Not imported by the inference service; needs inference/requirements-gpu.txt
(torch, diffusers, transformers, accelerate, safetensors) plus `peft`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("studio.train_lora")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class LoraSpec:
    name: str
    output: str
    instance_prompt: str
    data_dir: str
    rank: int = 16
    steps: int = 800
    learning_rate: float = 1e-4
    resolution: int = 1024

    @classmethod
    def from_dict(cls, d: dict) -> "LoraSpec":
        return cls(
            name=d["name"],
            output=d["output"],
            instance_prompt=d["instance_prompt"],
            data_dir=d["data_dir"],
            rank=int(d.get("rank", 16)),
            steps=int(d.get("steps", 800)),
            learning_rate=float(d.get("learning_rate", 1e-4)),
            resolution=int(d.get("resolution", 1024)),
        )


def train(spec: LoraSpec, base_model: str, out_dir: Path, repo_root: Path) -> Path:
    """Train one LoRA and write ``out_dir/<spec.output>``. Returns the path."""
    import torch
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from peft import LoraConfig, get_peft_model_state_dict
    from torchvision import transforms
    from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires CUDA.")

    data_dir = repo_root / spec.data_dir
    images = sorted(p for p in data_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"No instance images in {data_dir} (add a few before training {spec.name}).")

    device, dtype = "cuda", torch.float16
    log.info("Training LoRA %s: %d images, rank %d, %d steps", spec.name, len(images), spec.rank, spec.steps)

    # --- load the frozen backbone ------------------------------------------
    tok1 = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer", use_fast=False)
    tok2 = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer_2", use_fast=False)
    te1 = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=dtype).to(device)
    te2 = CLIPTextModelWithProjection.from_pretrained(base_model, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.float32).to(device)  # fp32: stable latents
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=dtype).to(device)
    noise_sched = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    for m in (te1, te2, vae, unet):
        m.requires_grad_(False)

    # --- attach LoRA to the UNet attention projections ----------------------
    unet.add_adapter(LoraConfig(
        r=spec.rank,
        lora_alpha=spec.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=spec.learning_rate)

    # --- precompute the (fixed) SDXL text conditioning ----------------------
    prompt_embeds, pooled = _encode_prompt(spec.instance_prompt, tok1, tok2, te1, te2, device, dtype)
    add_time_ids = torch.tensor(
        [[spec.resolution, spec.resolution, 0, 0, spec.resolution, spec.resolution]], device=device, dtype=dtype
    )

    to_tensor = transforms.Compose([
        transforms.Resize(spec.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(spec.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    latents = _encode_images(images, to_tensor, vae, device)  # [N,4,h,w] in fp16

    # --- training loop ------------------------------------------------------
    unet.train()
    for step in range(1, spec.steps + 1):
        idx = torch.randint(0, latents.shape[0], (1,), device=device)
        lat = latents[idx]
        noise = torch.randn_like(lat)
        t = torch.randint(0, noise_sched.config.num_train_timesteps, (1,), device=device).long()
        noisy = noise_sched.add_noise(lat, noise, t)

        pred = unet(
            noisy, t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
        ).sample
        target = noise if noise_sched.config.prediction_type == "epsilon" else \
            noise_sched.get_velocity(lat, noise, t)

        loss = torch.nn.functional.mse_loss(pred.float(), target.float())
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 100 == 0 or step == spec.steps:
            log.info("  %s step %d/%d loss=%.4f", spec.name, step, spec.steps, loss.item())

    # --- export in the diffusers LoRA layout --------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / spec.output
    lora_state = get_peft_model_state_dict(unet)
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(out_dir),
        unet_lora_layers=lora_state,
        weight_name=spec.output,
        safe_serialization=True,
    )
    log.info("Wrote %s", out_path)
    return out_path


def _encode_prompt(prompt, tok1, tok2, te1, te2, device, dtype):
    """SDXL uses two text encoders; concat hidden states, take pooled from the 2nd."""
    import torch

    embeds = []
    for tok, te in ((tok1, te1), (tok2, te2)):
        ids = tok(prompt, padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(device)
        out = te(ids, output_hidden_states=True)
        pooled = out[0]                       # only meaningful for te2 (projection head)
        embeds.append(out.hidden_states[-2])  # penultimate layer, per SDXL
    prompt_embeds = torch.cat(embeds, dim=-1).to(dtype)
    return prompt_embeds, pooled.to(dtype)


def _encode_images(images, to_tensor, vae, device):
    """VAE-encode instance images into the latent space once, up front."""
    import torch
    from PIL import Image

    pixels = torch.stack([to_tensor(Image.open(p).convert("RGB")) for p in images]).to(device, torch.float32)
    with torch.no_grad():
        latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
    return latents.to(torch.float16)
