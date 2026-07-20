<a id="readme-top"></a>

<div align="center">
  <h1>ptq-gpu</h1>
  <p>Quantised SDXL, served as TensorRT engines — with honest, side-by-side latency, VRAM, and quality metrics for every precision.</p>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=ffffff" alt="Python" /></a>
  <a href="https://developer.nvidia.com/tensorrt">
    <img src="https://img.shields.io/badge/TensorRT-10-76B900?style=flat-square&logo=nvidia&logoColor=ffffff" alt="TensorRT" /></a>
  <a href="https://nextjs.org/">
    <img src="https://img.shields.io/badge/Next.js-16-000000?style=flat-square&logo=nextdotjs&logoColor=ffffff" alt="Next.js" /></a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/aqilmarwan/ptq-gpu?style=flat-square" alt="last update" /></a>
  <h4>
    <a href="https://ptq-gpu.vercel.com">Live demo</a>
  </h4>
</div>

> [!NOTE]
> The **demo plane** runs the entire studio on CPU with **zero GPUs** — `docker compose up` and go. GPUs are only needed to build engines and to serve the real (TensorRT) plane.

> [!WARNING]
> The real TensorRT path (engine build + execution) is validated on-device; the demo plane is what CI exercises. FP8 currently builds as FP16 pending ModelOpt calibration.

# Table of contents

[Overview](#overview)

[How it works](#how-it-works)
- [Two planes](#two-planes)
- [Serving](#serving)
- [Build pipeline](#build-pipeline)
- [Infrastructure](#infrastructure)

[Variants](#variants)

[Running locally](#running-locally)
- [Requirements](#requirements)
- [Demo plane](#demo-plane)
- [Tests](#tests)

[Building engines](#building-engines)
- [On Modal](#on-modal)
- [On a GPU box](#on-a-gpu-box)
- [LoRA](#lora)

[Deployment](#deployment)

[Repository layout](#repository-layout)

[License](#license)

[Authors](#authors)

[Credits](#credits)

<p align="right"><a href="#readme-top">back to top</a></p>

---

## Overview

**ptq-gpu** is an image-generation studio built to make the cost of quantisation
*visible*. It serves the same SDXL 1.0 model as a matrix of **precision × style**
variants — FP16, INT8, and FP8, each as Base or LoRA — and reports the honest
trade-off for each one: on-disk size, peak VRAM, denoising throughput, and a
CLIP-based quality score, measured around the real inference loop.

Every variant is a prebuilt **TensorRT engine**. There is no Hugging Face or
`diffusers` at serving time: the text encoders, UNet, and VAE run as `.plan`
engines, and the scheduler is vendored. The web app (studio + compare pages)
streams generations over SSE and renders the metrics side by side.

<p align="right"><a href="#readme-top">back to top</a></p>

---

## How it works

### Two planes

The service picks its backend at startup, so the exact same product runs with or
without a GPU:

| Plane | When | Backend | GPU |
| --- | --- | --- | --- |
| **demo** | no CUDA, or `STUDIO_DEMO=1` | seeded procedural renderer, simulated metrics | none |
| **real** | CUDA + TensorRT present | prebuilt TRT engines, measured metrics | yes |

The demo plane makes the whole frontend exercisable in local dev and CI; its
metrics are derived from the registry and clearly logged as simulated.

### Serving

`inference/` is a FastAPI service exposing three endpoints — `GET /variants`,
`POST /generate` (SSE), and `GET /healthz`. `TensorRTBackend` keeps an LRU set of
engine bundles hot in VRAM, runs the denoise loop, and measures latency / VRAM /
throughput honestly. Engine bundles are synced from S3 into the pod by an init
container; nothing is downloaded from Hugging Face at request time.

### Build pipeline

`pipelines/` turns the base checkpoint into the servable engines, offline:

```
HF weights ──▶ ONNX export ──▶ TensorRT engine ──▶ publish to S3
   (once)      build_engines.py       (fp16/int8/fp8)      │
                                                            ▼
                                        serving pod syncs bundles → /engines
```

Metaflow orchestrates (`train LoRA → build engines → benchmark → register`), Ray
fans the per-variant work across GPUs, and MLflow records it. It runs on a
self-managed GPU box **or** serverless on Modal (`pipelines/modal_app.py`).

### Infrastructure

`infra/` provisions an EKS cluster with an L40S (FP8-capable) GPU node group for
serving, an ALB ingress terminating TLS for the inference API, and an HPA that
scales 1→3 replicas. The web app is hosted on Vercel and calls the API over HTTPS.

<p align="right"><a href="#readme-top">back to top</a></p>

---

## Variants

The precision × style matrix served by `GET /variants` (representative L40S
numbers; the build flow measures and syncs these into `inference/variants.yaml`):

| Variant | Precision | Style | Size | Peak VRAM | Throughput | Quality |
| --- | --- | --- | --- | --- | --- | --- |
| FP16 · Base | FP16 | Base | 13.0 GB | 18.4 GB | 7.8 it/s | 98 |
| INT8 · Base | INT8 | Base | 7.1 GB | 11.2 GB | 12.6 it/s | 95 |
| FP8 · Base | FP8 | Base | 6.6 GB | 9.6 GB | 16.4 it/s | 92 |
| FP16 · LoRA | FP16 | LoRA | 13.2 GB | 18.7 GB | 7.6 it/s | 97 |
| FP8 · LoRA | FP8 | LoRA | 6.8 GB | 9.9 GB | 15.8 it/s | 90 |

> [!NOTE]
> Only the UNet is quantised; the VAE and text encoders stay FP16. `quality` is a
> CLIP image-text score normalised against FP16 — i.e. *fidelity retained vs FP16*.

<p align="right"><a href="#readme-top">back to top</a></p>

---

## Running locally

### Requirements

- Docker (for the one-command demo), or Python 3.11 + Node 20 / pnpm for dev.
- **No GPU and no model downloads** are needed for anything in this section.

### Demo plane

The fastest path — inference (demo) + web, on CPU:

```bash
docker compose -f docker/docker-compose.yml up --build
# open http://localhost:3000
```

Or run the two services directly:

```bash
# inference (demo plane)
cd inference && pip install -r requirements.txt
STUDIO_DEMO=1 uvicorn main:app --port 8000

# web
cd web && pnpm install && pnpm dev   # http://localhost:3000
```

If the API is unreachable, the studio transparently falls back to in-browser demo
data, so the frontend is always usable.

### Tests

```bash
cd inference && STUDIO_DEMO=1 pytest -q     # API + smoke tests (demo plane)
cd web && pnpm lint && pnpm build           # lint + typecheck + build
```

<p align="right"><a href="#readme-top">back to top</a></p>

---

## Building engines

The real plane serves prebuilt TensorRT engines. Build them once on a GPU
(L40S for FP8), publish to S3, and point the serving pod at the bucket.

### On Modal

Serverless GPU — nothing to manage, per-second billing:

```bash
pip install modal && modal setup
modal secret create aws AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=<region>
modal volume create ptq-gpu-artifacts

# validate one variant first, then build them all
modal run pipelines/modal_app.py --s3-uri s3://<bucket> --skip-train --only fp16-base
modal run pipelines/modal_app.py --s3-uri s3://<bucket> --skip-train
```

### On a GPU box

```bash
pip install -r pipelines/requirements.txt -r inference/requirements.txt -r inference/requirements-gpu.txt
python pipelines/build_flow.py run --sync --engine-s3 s3://<bucket>
```

### LoRA

The LoRA is fused into the engine at build time, so a **pre-trained** SDXL LoRA
works with no dataset and no training:

```bash
./pipelines/fetch_lora.sh   # -> inference/loras/neon-atlas.safetensors
# then build with --skip-train (Modal), or plain build_flow on a GPU box
```

To train your own instead, add instance images under `pipelines/data/<name>/`
(see `pipelines/data/README.md`) and drop `--skip-train`.

<p align="right"><a href="#readme-top">back to top</a></p>

---

## Deployment

CI (`.github/workflows/ci.yml`) runs **test → build → push ECR → deploy EKS**.
Cluster bring-up (node groups, controllers, ingress, DNS/TLS) is documented in
[`infra/README.md`](infra/README.md). The short version:

```bash
eksctl create cluster -f infra/eksctl-cluster.yaml   # cluster + L40S GPU nodes + IRSA
infra/bootstrap.sh                                   # metrics-server, ALB controller, autoscaler
kubectl apply -f infra/k8s/                          # namespace, deployments, services, HPA, ingress
```

The inference pod becomes Ready once engines exist in the S3 bucket; until then,
set `STUDIO_DEMO=1` on the deployment to serve the demo plane with no GPU.

<p align="right"><a href="#readme-top">back to top</a></p>

---

## Repository layout

```text
ptq-gpu/
├── web/                      # Next.js studio + compare pages (hosted on Vercel)
├── inference/                # FastAPI service — serves prebuilt TensorRT engines (HF-free)
│   ├── main.py               #   /generate (SSE), /variants, /healthz
│   ├── pipelines.py          #   TensorRTBackend (engine LRU cache) + DemoBackend
│   ├── trt_runtime.py        #   TRT engine exec + vendored Euler scheduler + tokenizers
│   ├── metrics.py            #   latency, VRAM, throughput
│   └── variants.yaml         #   FP16 / INT8 / FP8 × Base / LoRA
├── pipelines/                # offline build/train (Metaflow + Ray + MLflow)
│   ├── build_flow.py         #   orchestration: train → build → benchmark → register
│   ├── build_engines.py      #   SDXL → ONNX → TensorRT engine → publish to S3
│   ├── train_lora.py         #   DreamBooth-LoRA (fused into the engine at build)
│   ├── modal_app.py          #   run the build/train on serverless Modal GPUs
│   └── fetch_lora.sh         #   grab a pre-trained LoRA (skip training)
├── infra/
│   ├── eksctl-cluster.yaml   # L40S (FP8) GPU node group + IRSA
│   ├── bootstrap.sh          # metrics-server, ALB controller, autoscaler
│   └── k8s/                  # deployments, services, HPA, ALB ingress (TLS)
├── .github/workflows/ci.yml  # test → build → push ECR → deploy EKS
└── docker/                   # Dockerfile.web, Dockerfile.inference, compose (local mode)
```

<p align="right"><a href="#readme-top">back to top</a></p>

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

## Authors

- **Aqil Marwan** — [@aqilmarwan](https://github.com/aqilmarwan)

## Credits

- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) — Stability AI
- [TensorRT](https://developer.nvidia.com/tensorrt) · [diffusers](https://github.com/huggingface/diffusers) · [FastAPI](https://fastapi.tiangolo.com/) · [Next.js](https://nextjs.org/) · [Modal](https://modal.com/)
- Cyberpunk SDXL LoRA — [issaccyj/lora-sdxl-cyberpunk](https://huggingface.co/issaccyj/lora-sdxl-cyberpunk)
- README format inspired by [vkayy/vkdb](https://github.com/vkayy/vkdb)

<p align="right"><a href="#readme-top">back to top</a></p>
