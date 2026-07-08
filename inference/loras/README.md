# loras/

LoRA weights referenced by `variants.yaml` (`serving.lora`) are loaded from this
directory at request time by `pipelines.DiffusersBackend` on the real plane.

The `.safetensors` files themselves are **not** committed (they're gitignored and
excluded from the Docker build) — they're produced by the build pipeline
(`pipelines/build_flow.py`) or downloaded from the model registry, and land here
by filename. For example `int4-lora` / `fp16-lora` expect:

```
loras/neon-atlas.safetensors
```

Behaviour when a referenced file is missing:

- **real plane** — the backend logs a warning and serves the base weights (no LoRA).
- **demo plane** — LoRA is ignored entirely; the seeded procedural renderer runs.

So the studio stays fully runnable locally without any weights present.
