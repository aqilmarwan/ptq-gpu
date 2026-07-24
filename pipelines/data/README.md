# LoRA training data

Instance images for the DreamBooth-LoRA training step (`train_lora.py`). These
are **your** images and are gitignored — only this README is tracked.

## Layout

One folder per LoRA, named to match the `name` in `pipelines/loras.yaml`:

```
pipelines/data/
└── neon-atlas/          # matches loras.yaml: name: neon-atlas
    ├── 01.png
    ├── 02.png
    └── ...
```

## What makes a good set

- **5–20 images** of the single concept/style you want to capture.
- Ideally **1024×1024** (or larger, square-ish); the trainer center-crops to
  `resolution` from `loras.yaml`.
- `.png`, `.jpg`, `.jpeg`, or `.webp`.
- Consistent subject/style, varied pose/background — that's what the
  `instance_prompt` in `loras.yaml` teaches the model to associate.

## How it's consumed

The build flow's `train` step (`train_lora.train()`) reads `pipelines/data/<name>/`
on the GPU box before building the LoRA engines.

## If you don't provide images

Nothing breaks. Training for that LoRA is skipped with a clear message, and the
LoRA engine variants (`fp16-lora`, `fp8-lora`) simply aren't built — the base
variants (`fp16-base`, `int8-base`, `fp8-base`) build fine on their own. Or drop
the LoRA variants from `variants.yaml` to build a clean base-only catalog.
