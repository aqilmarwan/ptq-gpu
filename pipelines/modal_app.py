"""Run the GPU build/train pipeline on Modal (serverless L40S).

Executes the same logic as `build_flow`'s real plane -- `train_lora.py` to train
the LoRAs and `build_engines.py` to compile + publish the TensorRT engines -- but
on Modal's on-demand GPUs instead of a self-managed box or an EKS GPU node. You
pay per second the job runs; nothing is left warm.

One-time setup (on your laptop -- only the Modal client is needed locally):
    pip install modal
    modal setup                                   # auth
    modal secret create aws \\
        AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=ap-southeast-5
    modal volume create ptq-gpu-artifacts         # shared data + LoRA weights
    modal volume put ptq-gpu-artifacts pipelines/data data   # upload instance images

Run:
    modal run pipelines/modal_app.py --s3-uri s3://quant-studio-engine-bucket
    modal run pipelines/modal_app.py --s3-uri s3://... --skip-train   # engines only

The AWS secret needs s3:PutObject on the engine bucket. SDXL base is open, so no
HF token is required; if you point BASE_MODEL at a gated model, attach a
`huggingface` secret (HF_TOKEN) to the two functions.
"""

import sys
from pathlib import Path

import modal

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"  # real HF id (variants.yaml carries a display name)
GPU = "L40S"  # Ada -> FP8 capable, 48GB
ENGINE_S3_URI = "s3://quant-studio-engine-bucket"  # where build_one publishes / serve pulls from

image = (
    # System CUDA from the NGC base so the torch + tensorrt wheels resolve their
    # GPU libs cleanly (more reliable than debian_slim, where the tensorrt wheel
    # can fail to find CUDA). add_python gives a clean 3.11 to install into.
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .pip_install(
        "torch>=2.4",
        # Pin to TRT 10.x: engines are NOT portable across major versions, and the
        # serving image (NGC tensorrt:24.08) is TRT 10.3. `>=10.0` resolves to
        # TRT 11, whose .plan files won't deserialize on the 10.x runtime.
        "tensorrt==10.3.0",
        "diffusers>=0.31",
        "transformers>=4.44",
        "accelerate>=0.33",
        "peft>=0.11",
        "torchvision>=0.19",
        "safetensors>=0.4",
        "onnx>=1.16",
        "onnxscript>=0.1",   # torch.onnx.export (dynamo path) needs this
        "boto3>=1.34",
        "pyyaml>=6.0",
        "pillow>=10.3",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # the build/train libraries (they don't import metaflow, so they load standalone)
    .add_local_dir("pipelines", "/root/pipelines")
)

app = modal.App("ptq-gpu-build", image=image)
artifacts = modal.Volume.from_name("ptq-gpu-artifacts", create_if_missing=True)

# Serving image: HF-free (no diffusers/peft/onnx) -- just the TRT runtime + the
# CLIP tokenizer + the FastAPI stack. tensorrt is pinned to the SAME 10.3 the
# engines were built with (plans aren't portable across versions).
serving_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .pip_install(
        "torch>=2.4",
        "tensorrt==10.3.0",
        "transformers>=4.44",
        "fastapi>=0.115",
        "uvicorn[standard]>=0.30",
        "pydantic>=2.7",
        "pyyaml>=6.0",
        "pillow>=10.3",
        "numpy<2.3",
        "boto3>=1.34",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_dir("inference", "/root/inference")
)


def _bootstrap() -> None:
    if "/root/pipelines" not in sys.path:
        sys.path.insert(0, "/root/pipelines")


@app.function(gpu=GPU, timeout=2 * 60 * 60, volumes={"/artifacts": artifacts})
def train_one(lora: dict) -> str:
    """Train one LoRA into the shared volume at /artifacts/loras/<name>.safetensors."""
    _bootstrap()
    from train_lora import LoraSpec, train

    spec = LoraSpec.from_dict(lora)
    spec.data_dir = f"data/{spec.name}"  # instance images live under the volume root
    train(spec, BASE_MODEL, Path("/artifacts/loras"), Path("/artifacts"))
    artifacts.commit()
    return spec.name


@app.function(
    gpu=GPU,
    timeout=60 * 60,
    volumes={"/artifacts": artifacts},
    secrets=[modal.Secret.from_name("aws")],
)
def build_one(variant: dict, s3_uri: str) -> str:
    """Build one variant's engine bundle and publish it to S3."""
    _bootstrap()
    from build_engines import EngineSpec, build_bundle, publish_s3

    sv = variant.get("serving", {}) or {}
    spec = EngineSpec(
        engine=sv.get("engine", variant["id"]),
        precision=sv.get("precision", "fp16"),
        lora=sv.get("lora"),
    )
    artifacts.reload()  # pick up any LoRA the train step committed
    bundle = build_bundle(spec, BASE_MODEL, Path("/tmp/engines"), Path("/artifacts/loras"))
    publish_s3(bundle, s3_uri.rstrip("/"))
    return spec.engine


def _sync_engines(s3_uri: str, dest: str) -> None:
    """Mirror every object under the bucket/prefix into ``dest`` (like ``aws s3 sync``)."""
    import boto3

    bucket, _, prefix = s3_uri[len("s3://"):].partition("/")
    prefix = prefix.rstrip("/")
    s3 = boto3.client("s3")
    kwargs = {"Bucket": bucket, **({"Prefix": prefix} if prefix else {})}
    count = 0
    for page in s3.get_paginator("list_objects_v2").paginate(**kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):].lstrip("/") if prefix else key
            local = Path(dest) / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local))
            count += 1
    print(f"synced {count} engine files from {s3_uri} -> {dest}")


@app.function(
    image=serving_image,
    gpu=GPU,
    secrets=[modal.Secret.from_name("aws")],
    min_containers=1,     # keep one L40S warm -> no cold starts (cost tradeoff, as requested)
    timeout=60 * 60,
)
@modal.asgi_app()
def serve():
    """Serve the real FastAPI inference app on a warm L40S; engines pulled from S3.

    Gives a public HTTPS URL with no EKS / ingress / cert / DNS. Point Vercel's
    NEXT_PUBLIC_API_URL at it. This is also the on-device test of trt_runtime.py.
    """
    import os

    os.environ.setdefault("STUDIO_ENGINE_DIR", "/engines")
    os.environ.setdefault("STUDIO_CORS_ORIGINS", "*")   # tighten to your Vercel origin later
    _sync_engines(ENGINE_S3_URI, "/engines")            # before importing main (Registry reads config at import)
    sys.path.insert(0, "/root/inference")
    from main import app as fastapi_app
    return fastapi_app


@app.local_entrypoint()
def main(s3_uri: str, skip_train: bool = False, only: str = ""):
    """Orchestrate: train LoRAs (unless --skip-train) -> build + publish engines.

    --only fp16-base           build just one variant (smoke-test the pipeline)
    --only fp16-base,int8-base  build a subset (comma-separated ids)
    """
    import yaml

    variants = yaml.safe_load(open("inference/variants.yaml"))["variants"]
    loras = (yaml.safe_load(open("pipelines/loras.yaml")) or {}).get("loras", [])

    if only:
        wanted = {v.strip() for v in only.split(",") if v.strip()}
        variants = [v for v in variants if v["id"] in wanted]
        # only train LoRAs the selected variants actually need
        needed = {v["serving"].get("lora") for v in variants if v.get("serving", {}).get("lora")}
        loras = [lo for lo in loras if lo["name"] in needed]
        if not variants:
            raise SystemExit(f"--only matched no variants; known ids: "
                             f"{[v['id'] for v in yaml.safe_load(open('inference/variants.yaml'))['variants']]}")

    if not skip_train and loras:
        print(f"Training {len(loras)} LoRA(s) on {GPU}...")
        for lora, res in zip(loras, train_one.map(loras, return_exceptions=True)):
            print(f"  train {lora['name']} FAILED: {res}" if isinstance(res, Exception)
                  else f"  trained {res}")

    print(f"Building {len(variants)} engine bundles on {GPU} -> {s3_uri}")
    for variant, res in zip(
        variants, build_one.map(variants, kwargs={"s3_uri": s3_uri}, return_exceptions=True)
    ):
        print(f"  build {variant['id']} FAILED: {res}" if isinstance(res, Exception)
              else f"  published {res}")

    print(f"\nDone. Set STUDIO_ENGINE_S3_URI={s3_uri} on the inference deployment.")
