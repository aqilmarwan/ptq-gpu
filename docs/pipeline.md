# ptq-gpu — pipeline diagrams

## 1. End-to-end pipeline (activity)

Offline **build** → **deploy** (CI/CD) → **serve** on a pinned GPU → **benchmark**.

```mermaid
flowchart TB
    subgraph build["Build pipeline — offline, GPU (Modal or GPU box), orchestrated by build_flow.py"]
        direction TB
        hf["SDXL 1.0 weights (Hugging Face)"]
        lora["LoRA .safetensors<br/>(trained or fetched)"]
        fuseLora["fuse LoRA into UNet<br/>(fuse_lora)"]
        onnx["export ONNX<br/>text_encoder x2 · unet · vae_decoder"]
        trt["build TensorRT engines<br/>fp16 · int8 (calibrated) · fp8"]
        bundle["bundle: .plan x4 + tokenizers + metadata"]
        s3[("S3<br/>engine bundles")]

        hf --> fuseLora
        lora -. optional .-> fuseLora
        fuseLora --> onnx --> trt --> bundle --> s3
    end

    subgraph deploy["Deploy pipeline — CI/CD (.github/workflows/ci.yml)"]
        direction TB
        push["git push (main)"]
        test["test: pytest (demo plane) + web lint/build"]
        image["build Dockerfile.inference (TRT 10.3)"]
        ecr[("ECR<br/>inference image")]
        rollout["kubectl set image + rollout"]

        push --> test --> image --> ecr --> rollout
    end

    subgraph serve["Serve pipeline — EKS Auto Mode, one GPU node"]
        direction TB
        init["init container: aws s3 sync to /engines"]
        pod["inference pod<br/>FastAPI + TensorRTBackend"]

        init --> pod
    end

    subgraph clients["Clients"]
        direction TB
        web["Web studio (Vercel)"]
        bench["scripts/bench.py"]
    end

    s3 --> init
    rollout --> pod
    web -->|"POST /generate (SSE)"| pod
    bench -->|"N requests @ concurrency"| pod
    pod -->|"p50 / p95 / p99"| bench
```

## 2. Request flow (sequence)

One `POST /generate` on the real (TensorRT) plane.

```mermaid
sequenceDiagram
    autonumber
    participant C as Client (web / bench.py)
    participant API as FastAPI (main.py)
    participant B as TensorRTBackend
    participant E as TRT engines (VRAM)
    participant S as EulerScheduler

    C->>API: POST /generate {prompt, variantId, steps}
    API->>B: run(params, emit)
    B->>B: _ensure_loaded(variant) - LRU, cold-load .plan from /engines
    B-->>C: SSE status: load
    B->>E: text_encoder x2 (tokenized prompt + negative)
    E-->>B: prompt_embeds + pooled
    B-->>C: SSE status: denoise
    loop for each step
        B->>S: scale_model_input(latents)
        B->>E: unet(sample, t, embeds) - CFG, batch 2
        E-->>B: noise_pred
        B->>S: step() -> latents
        B-->>C: SSE progress {step, totalSteps}
    end
    B-->>C: SSE status: decode
    B->>E: vae_decoder(latents / scale)
    E-->>B: image tensor [-1, 1]
    B-->>C: SSE done {imageUrl (base64), metrics}
```

## 3. Backend selection (state)

The service picks its plane at startup — the same product with or without a GPU.

```mermaid
stateDiagram-v2
    [*] --> Startup
    Startup --> Demo: STUDIO_DEMO=1<br/>or no CUDA<br/>or no TensorRT
    Startup --> TensorRT: CUDA + TensorRT present
    Demo: DemoBackend<br/>(procedural image, simulated metrics)
    TensorRT: TensorRTBackend<br/>(prebuilt engines, measured metrics)
    Demo --> [*]
    TensorRT --> [*]
```
