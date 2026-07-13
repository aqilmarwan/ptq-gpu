# pipelines — build & benchmark flow

`build_flow.py` is the offline pipeline that turns the base SDXL checkpoint into
the servable **TensorRT engine bundles** (`build_engines.py`: ONNX export →
engine build → publish to S3) and measures the `registry:` metrics that
`inference/variants.yaml` surfaces through `GET /variants`. This is the only
place Hugging Face is used — the serving plane runs the engines HF-free.

- **Metaflow** orchestrates: train LoRA → build engines → benchmark → register.
- **Ray** fans the per-variant engine build + benchmark across the cluster's GPUs.
- **MLflow** logs params, measured metrics, and registers each variant.
- Benchmarks run the freshly built engines through the *serving* backend, so the
  numbers come from the exact engines that will serve traffic.

It shares the inference service's two-plane design: a real CUDA plane that builds
and measures honestly, and a `--demo` plane that simulates from the catalog
baseline so the flow runs on a laptop or in CI with no GPU or downloads.

## Run

```bash
pip install -r pipelines/requirements.txt

# laptop / CI — simulated, no GPU, does not touch variants.yaml
python pipelines/build_flow.py run --demo

# real GPU box — build, benchmark, and write measured metrics back
pip install -r inference/requirements.txt -r inference/requirements-gpu.txt
python pipelines/build_flow.py run --trials 5 --sync
```

### On Modal (serverless GPU — no box to manage)

`modal_app.py` runs the same `train_lora` + `build_engines` on an on-demand L40S
and publishes engines straight to S3. Only the Modal client is needed locally.

```bash
pip install modal && modal setup
modal secret create aws AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=ap-southeast-5
modal volume create ptq-gpu-artifacts

# Skip training: grab a pre-trained LoRA, push it to the volume, build everything.
./pipelines/fetch_lora.sh
modal volume put ptq-gpu-artifacts inference/loras/neon-atlas.safetensors loras/neon-atlas.safetensors
modal run pipelines/modal_app.py --s3-uri s3://quant-studio-engine-bucket --skip-train

# Or validate one variant first:
modal run pipelines/modal_app.py --s3-uri s3://... --skip-train --only fp16-base
```

`--skip-train` skips the training step but still builds **all** variants,
including the LoRA ones, from the fetched weights. To train the LoRA yourself
instead, upload instance images (`pipelines/data/`) and drop `--skip-train`.

## Outputs

- `pipelines/out/registry.json` — measured metrics per variant (always written).
- `inference/variants.yaml` — `registry:` blocks updated in place, comments
  preserved (only with `--sync`).
- MLflow experiment `ptq-gpu/sdxl-quant` — one parent run per build, one nested
  run per variant.

## What's measured (real plane)

- **size / VRAM / throughput** — measured around a real denoise loop.
- **quality** — a CLIP image-text score over a fixed prompt set, normalised
  against `fp16-base` so it reads as *fidelity retained vs FP16* (FP16 ≈ 100).
- **LoRAs** — the `train` step trains adapters declared in `loras.yaml` into
  `inference/loras/` before benchmarking (real plane only; `--demo` skips it).

The `--demo` plane simulates all of the above from the catalog baseline.
