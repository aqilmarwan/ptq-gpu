# pipelines — build & benchmark flow

`build_flow.py` is the offline pipeline that turns the base SDXL checkpoint into
the servable, quantised variants and measures the `registry:` metrics that
`inference/variants.yaml` surfaces through `GET /variants`.

- **Metaflow** orchestrates and versions each run (`run`, `resume`, `show`).
- **Ray** fans the per-variant build + benchmark across the cluster's GPUs.
- **MLflow** logs params, measured metrics, and registers each variant.

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

## Outputs

- `pipelines/out/registry.json` — measured metrics per variant (always written).
- `inference/variants.yaml` — `registry:` blocks updated in place, comments
  preserved (only with `--sync`).
- MLflow experiment `ptq-gpu/sdxl-quant` — one parent run per build, one nested
  run per variant.

> Quality (CLIP/aesthetic) scoring is not yet wired; the flow carries the catalog
> baseline for `quality` rather than emitting an unmeasured number. Size, VRAM,
> and throughput are measured.
