# loras/

LoRA `.safetensors` trained by the build pipeline (`pipelines/train_lora.py`)
land here by filename, e.g.:

```
loras/neon-atlas.safetensors
```

They are **build-time** inputs, not serving-time ones: `pipelines/build_engines.py`
fuses the LoRA into the base UNet (`fuse_lora`) before exporting to ONNX, so the
style is baked into the `fp16-lora` / `fp8-lora` TensorRT engines. The serving
plane loads engines only — it never reads this directory.

## Fastest way to get one: reuse a pre-trained LoRA (no training)

Because the LoRA is fused in, a ready-made SDXL LoRA works as-is — no dataset, no
training GPU. Grab one and build the LoRA variants with `--skip-train`:

```bash
./pipelines/fetch_lora.sh   # -> inference/loras/neon-atlas.safetensors (cyberpunk SDXL LoRA)
```

Then build (`--skip-train` skips training but still builds every variant,
including the LoRA ones, from the fetched weights).

The files are gitignored and excluded from the Docker build. If a LoRA is missing
at engine-build time the build fails for that variant (train it first); the demo
plane needs no weights at all and runs the seeded procedural renderer.
