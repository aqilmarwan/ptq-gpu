# ptq-gpu

## File Directory Structure

ptq-gpu/
├── README.md                # vision, screenshots, quant tradeoff table, how to run, cost notes
├── web/                     # Next.js app (studio page + compare page)
├── inference/               # FastAPI service — serves prebuilt TensorRT engines (HF-free)
│   ├── main.py              # FastAPI: /generate (SSE), /variants, /healthz
│   ├── pipelines.py         # TensorRTBackend (engine LRU cache) + DemoBackend
│   ├── trt_runtime.py       # TRT engine exec + vendored Euler scheduler + tokenizers
│   ├── metrics.py           # latency, VRAM, throughput
│   └── variants.yaml        # FP16 / INT8 / FP8 × Base / LoRA
├── pipelines/               # offline build (Metaflow + Ray + MLflow)
│   ├── build_flow.py        # orchestration: train LoRA → build engines → benchmark → register
│   ├── build_engines.py     # SDXL → ONNX → TensorRT engine → publish to S3
│   └── train_lora.py        # DreamBooth-LoRA (fused into the engine at build)
├── infra/
│   ├── eksctl-cluster.yaml  # L40S (FP8) GPU node group + IRSA
│   ├── bootstrap.sh         # metrics-server, ALB controller, autoscaler
│   └── k8s/                 # deployments, services, HPA, ALB ingress (TLS)
├── .github/workflows/ci.yml # test → build → push ECR → deploy EKS
└── docker/                  # Dockerfile.web, Dockerfile.inference, compose (local mode)