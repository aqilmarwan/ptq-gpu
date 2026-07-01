# inference — Quant Studio FastAPI service

Real diffusion inference behind two endpoints the frontend consumes:

| Endpoint          | Returns                                                        |
| ----------------- | ------------------------------------------------------------- |
| `GET  /variants`  | the precision x style matrix + registered benchmark metrics   |
| `POST /generate`  | `text/event-stream` — `status` / `progress` / `done` events   |
| `GET  /healthz`   | liveness + which plane is active                              |

## Two planes

* **cuda** — real SDXL on a GPU. Lazily loads each variant, keeps an LRU set hot
  in VRAM (`max_resident` in `variants.yaml`), quantises per variant, and
  measures latency / VRAM / throughput **around the real denoise loop**.
* **demo** — no GPU, no model download. Seeded procedural image + simulated
  timing, so the frontend is fully exercisable locally. Metrics are simulated
  and logged as such. Selected automatically when no CUDA device is present.

## Run

Demo plane (local, no GPU — just the core deps):

```bash
cd inference
pip install -r requirements.txt
STUDIO_DEMO=1 uvicorn main:app --port 8000 --reload
```

Real plane (GPU box):

```bash
pip install -r requirements.txt -r requirements-gpu.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Point the frontend at it with `NEXT_PUBLIC_API_URL=http://localhost:8000`.

## Variant matrix

Defined in `variants.yaml` — `serving:` says how to load each (quant + LoRA),
`registry:` carries the benchmarked display metrics. Adding a variant is a
config edit. Precision via `optimum-quanto` (INT8 / INT4 weight-only on the
UNet; VAE + text encoders stay FP16). LoRA weights go in `loras/`.

## SSE event shapes (`schemas.py` / `web/lib/types.ts`)

```jsonc
{ "type": "status",   "stage": "load|denoise|decode", "message": "…", "cold": true }
{ "type": "progress", "step": 12, "totalSteps": 30 }
{ "type": "done",     "result": { "imageUrl": "data:image/png;base64,…",
                                   "variantId": "…", "params": {}, "metrics": {} } }
{ "type": "error",    "message": "…" }
```

## Config (env)

| Var                    | Default                          | Purpose                         |
| ---------------------- | -------------------------------- | ------------------------------- |
| `STUDIO_DEMO`          | unset                            | force the demo plane            |
| `STUDIO_CORS_ORIGINS`  | `http://localhost:3000,…:3100`   | allowed browser origins         |
| `STUDIO_VARIANTS`      | `./variants.yaml`                | variant catalog path            |
| `STUDIO_LORA_DIR`      | `./loras`                        | LoRA weights directory          |
| `STUDIO_BASE_MODEL`    | from `variants.yaml`             | override base checkpoint        |
