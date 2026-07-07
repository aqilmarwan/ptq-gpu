"""Demo-plane smoke test — validates the service boots and generates end-to-end
on CPU with no model download, exercising config, the variant catalog, schemas,
and the full `Registry.run` path."""

from pipelines import Registry
from schemas import GenerationParams


def _build_registry() -> Registry:
    reg = Registry()
    assert reg.plane == "demo"
    return reg


def test_variant_catalog_loads():
    variants = _build_registry().variants()
    assert variants, "variants.yaml produced no variants"
    # every variant the frontend can pick must be routable
    reg = _build_registry()
    assert all(reg.has(v.id) for v in variants)


def test_demo_generation_end_to_end():
    reg = _build_registry()
    variant = reg.variants()[0]

    events: list[dict] = []
    params = GenerationParams(
        prompt="a smoke test",
        variant_id=variant.id,
        steps=2,  # keep the simulated denoise loop fast in CI
        seed=7,
        width=256,
        height=256,
    )
    result = reg.run(params, events.append)

    assert result.variant_id == variant.id
    assert result.image_url.startswith("data:image/png;base64,")
    assert result.metrics.latency_ms.total > 0
    # the SSE stream should have covered load -> denoise -> decode
    stages = {e.get("stage") for e in events if e["type"] == "status"}
    assert {"load", "denoise", "decode"} <= stages
