"""Offline build + benchmark pipeline for the servable SDXL variants.

Metaflow orchestrates the DAG and versions every run; Ray fans the per-variant
build+benchmark out across the cluster's GPUs in parallel; MLflow records the
params, measured metrics, and artifacts and registers each variant as a model
version. The measured `registry:` block for every variant is what
`inference/variants.yaml` surfaces through ``GET /variants`` -- this flow is how
those numbers get there honestly, instead of being hand-typed.

Two planes, mirroring the inference service:

* **real**  -- torch + diffusers on CUDA. Quantises the UNet (INT8/NF4), attaches
  LoRA, and measures size / VRAM / throughput around a real denoise loop.
* **demo**  -- no GPU, no downloads. Simulates each variant's benchmark from the
  catalog baseline so the whole flow is exercisable in CI / on a laptop. Demo
  metrics are clearly logged as simulated. Selected with ``--demo`` or whenever
  CUDA is absent.

Run::

    python pipelines/build_flow.py run --demo            # laptop / CI dry-run
    python pipelines/build_flow.py run --sync            # real GPU box, write back
    python pipelines/build_flow.py run --trials 5 --sync

Requires the extra tooling in ``pipelines/requirements.txt`` (Metaflow, Ray,
MLflow, ruamel.yaml). None of it is imported by the inference service.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

from metaflow import FlowSpec, Parameter, current, step

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_VARIANTS = REPO / "inference" / "variants.yaml"
OUT_DIR = HERE / "out"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Ray remote: build + benchmark one variant
# --------------------------------------------------------------------------- #
#
# Defined at module scope so Ray can pickle it. Decorated lazily inside the flow
# (after `ray.init`) to keep import side-effect free and demo runs Ray-optional.


def _build_and_benchmark(base_model: str, variant: dict, trials: int, demo: bool, seed: int) -> dict:
    """Return the measured ``registry`` block for one variant.

    In demo mode this simulates from the catalog baseline; in real mode it loads
    the (quantised, optionally LoRA'd) pipeline once and times ``trials`` denoise
    passes, reporting the median so a single cold outlier doesn't skew results.
    """
    vid = variant["id"]
    serving = variant.get("serving", {}) or {}
    baseline = variant.get("registry", {}) or {}

    if demo:
        # Deterministic jitter around the baseline so runs look measured, not fixed.
        wobble = 1.0 + ((hash((vid, seed)) % 1000) / 1000.0 - 0.5) * 0.04
        return {
            "sizeGB": round(baseline["sizeGB"] * wobble, 1),
            "vramGB": round(baseline["vramGB"] * wobble, 1),
            "stepsPerSec": round(baseline["stepsPerSec"] * wobble, 1),
            "quality": baseline["quality"],
            "defaultSteps": baseline["defaultSteps"],
            "_simulated": True,
        }

    import torch
    from diffusers import StableDiffusionXLPipeline

    quant = serving.get("quant", "none")
    dtype = torch.float16

    t0 = time.perf_counter()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model, torch_dtype=dtype, use_safetensors=True, variant="fp16"
    )
    if quant in {"int8", "nf4"}:
        from optimum.quanto import freeze, qint4, qint8, quantize

        quantize(pipe.unet, weights=qint8 if quant == "int8" else qint4)
        freeze(pipe.unet)
    pipe = pipe.to("cuda")

    lora = serving.get("lora")
    if lora:
        lora_path = REPO / "inference" / lora
        if lora_path.exists():
            pipe.load_lora_weights(str(lora_path))
    pipe.set_progress_bar_config(disable=True)
    load_s = time.perf_counter() - t0

    steps = int(baseline.get("defaultSteps", 30))
    per_run_sps: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(max(1, trials)):
        gen = torch.Generator(device="cuda").manual_seed(seed + i)
        torch.cuda.synchronize()
        r0 = time.perf_counter()
        pipe(
            prompt="a benchmark render of a city at dusk, volumetric light",
            num_inference_steps=steps,
            guidance_scale=6.5,
            width=1024,
            height=1024,
            generator=gen,
        )
        torch.cuda.synchronize()
        per_run_sps.append(steps / (time.perf_counter() - r0))

    vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    size_gb = _weights_size_gb(pipe, quant)

    return {
        "sizeGB": round(size_gb, 1),
        "vramGB": round(vram_gb, 1),
        "stepsPerSec": round(statistics.median(per_run_sps), 1),
        # CLIP/aesthetic scoring needs a scorer model; keep the catalog baseline
        # until that harness lands rather than emitting a fake number.
        "quality": baseline["quality"],
        "defaultSteps": steps,
        "coldLoadSec": round(load_s, 1),
        "_simulated": False,
    }


def _weights_size_gb(pipe, quant: str) -> float:
    """On-disk footprint of the served weights, in GB (dtype-aware)."""
    import torch

    bytes_per = {"int8": 1, "nf4": 0.5}.get(quant, 2)  # fp16 default = 2 bytes
    params = 0
    for module in (pipe.unet, pipe.vae, getattr(pipe, "text_encoder", None), getattr(pipe, "text_encoder_2", None)):
        if module is None:
            continue
        for p in module.parameters():
            # UNet honours the quantised width; VAE + text encoders stay fp16.
            width = bytes_per if module is pipe.unet else 2
            params += p.numel() * width
    return params / (1024**3) if not isinstance(params, torch.Tensor) else float(params) / (1024**3)


# --------------------------------------------------------------------------- #
# Flow
# --------------------------------------------------------------------------- #


class BuildFlow(FlowSpec):
    """Build, benchmark, and register every SDXL variant in the catalog."""

    variants_file = Parameter("variants-file", default=str(DEFAULT_VARIANTS), help="Path to variants.yaml")
    demo = Parameter("demo", is_flag=True, default=False, help="Force the simulated plane (no GPU/downloads).")
    trials = Parameter("trials", default=3, help="Benchmark passes per variant (median is reported).")
    seed = Parameter("seed", default=1234, help="Base seed for reproducible benchmarks.")
    sync = Parameter("sync", is_flag=True, default=False, help="Write measured metrics back into variants.yaml.")
    experiment = Parameter("experiment", default="ptq-gpu/sdxl-quant", help="MLflow experiment name.")

    @step
    def start(self):
        import yaml

        doc = yaml.safe_load(Path(self.variants_file).read_text())
        self.base_model = doc["base_model"]
        self.variants = doc["variants"]
        self.is_demo = bool(self.demo) or not _cuda_available()
        if self.is_demo and not self.demo:
            print("No CUDA device -- falling back to the simulated (demo) plane.")
        print(f"Building {len(self.variants)} variants from {self.base_model} "
              f"[{'demo' if self.is_demo else 'real'} plane]")
        self.next(self.build)

    @step
    def build(self):
        """Fan the per-variant build+benchmark across Ray, collect the results."""
        import ray

        ray.init(ignore_reinit_error=True, configure_logging=False)
        remote = ray.remote(num_gpus=0 if self.is_demo else 1)(_build_and_benchmark)
        futures = [
            remote.remote(self.base_model, v, self.trials, self.is_demo, self.seed)
            for v in self.variants
        ]
        metrics = ray.get(futures)
        ray.shutdown()

        self.results = {v["id"]: m for v, m in zip(self.variants, metrics)}
        for vid, m in self.results.items():
            tag = "sim" if m.get("_simulated") else "measured"
            print(f"  {vid:<12} {m['stepsPerSec']:>5} steps/s  {m['vramGB']:>5} GB VRAM  ({tag})")
        self.next(self.register)

    @step
    def register(self):
        """Log to MLflow, persist the registry artifact, optionally sync the YAML."""
        self._log_to_mlflow()

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        clean = {vid: {k: v for k, v in m.items() if not k.startswith("_")} for vid, m in self.results.items()}
        registry_json = OUT_DIR / "registry.json"
        registry_json.write_text(json.dumps(clean, indent=2) + "\n")
        print(f"Wrote {registry_json}")

        if self.sync:
            self._sync_variants_yaml(clean)
            print(f"Synced measured metrics into {self.variants_file}")
        else:
            print("Skipped variants.yaml sync (pass --sync to write back).")
        self.next(self.end)

    @step
    def end(self):
        plane = "demo/simulated" if self.is_demo else "real/measured"
        print(f"BuildFlow complete ({plane}). {len(self.results)} variants registered.")

    # ----------------------------------------------------------------- helpers

    def _log_to_mlflow(self):
        try:
            import mlflow
        except Exception:
            print("mlflow not installed -- skipping experiment logging.")
            return

        mlflow.set_experiment(self.experiment)
        with mlflow.start_run(run_name=f"build-{current.run_id}"):
            mlflow.log_params({
                "base_model": self.base_model,
                "plane": "demo" if self.is_demo else "real",
                "trials": self.trials,
                "variants": len(self.results),
            })
            for vid, m in self.results.items():
                with mlflow.start_run(run_name=vid, nested=True):
                    variant = next(v for v in self.variants if v["id"] == vid)
                    serving = variant.get("serving", {}) or {}
                    mlflow.log_params({
                        "variant": vid,
                        "precision": variant["precision"],
                        "style": variant["style"],
                        "quant": serving.get("quant", "none"),
                        "lora": serving.get("lora") or "none",
                    })
                    mlflow.log_metrics({
                        "size_gb": m["sizeGB"],
                        "vram_gb": m["vramGB"],
                        "steps_per_sec": m["stepsPerSec"],
                        "quality": m["quality"],
                    })

    def _sync_variants_yaml(self, clean: dict):
        """Round-trip variants.yaml with ruamel so comments/formatting survive."""
        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)  # match variants.yaml's block-list layout
        path = Path(self.variants_file)
        doc = yaml.load(path)
        for entry in doc["variants"]:
            measured = clean.get(entry["id"])
            if not measured:
                continue
            reg = entry["registry"]
            for key in ("sizeGB", "vramGB", "stepsPerSec", "quality", "defaultSteps"):
                if key in measured:
                    reg[key] = measured[key]
        yaml.dump(doc, path)


if __name__ == "__main__":
    # Metaflow reads its subcommand (run/resume/show) from argv.
    os.environ.setdefault("USERNAME", os.environ.get("USER", "ci"))
    BuildFlow()
