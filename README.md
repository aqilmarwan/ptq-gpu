# ptq-gpu

## File Directory Structure

ptq-gpu/
├── README.md                # vision, screenshots, quant tradeoff table, how to run, cost notes
├── web/                     # Next.js app (studio page + compare page)
├── inference/
│   ├── main.py              # FastAPI: /generate (SSE), /variants
│   ├── pipelines.py         # load/cache/hot-swap variants
│   ├── metrics.py           # latency, VRAM, throughput
│   └── variants.yaml
├── pipelines/
│   └── build_flow.py        # Metaflow + Ray-parallel build/LoRA + MLflow logging
├── infra/
│   ├── eksctl-cluster.yaml  # GPU node group
│   └── k8s/                 # deployment.yaml, service.yaml, hpa.yaml
├── .github/workflows/ci.yml # test → build → push ECR → deploy EKS
└── docker/                  # Dockerfile.web, Dockerfile.inference, compose (local mode)