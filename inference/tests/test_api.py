"""HTTP-layer tests for main.py on the demo plane (no GPU, no engines).

Exercises the FastAPI contract the frontend depends on: the variant catalog
(camelCase + precision set), the /generate SSE stream (stage order, terminal
result, camelCase metrics), the server-side guard-rails, and the error path.
Runs in CI because conftest forces STUDIO_DEMO."""

import json

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


def _sse_events(resp) -> list[dict]:
    """Parse an SSE response body into the list of decoded event dicts."""
    events: list[dict] = []
    buf = ""
    for chunk in resp.iter_text():
        buf += chunk
        while "\n\n" in buf:
            frame, buf = buf.split("\n\n", 1)
            data = "".join(l[5:].strip() for l in frame.splitlines() if l.startswith("data:"))
            if data:
                events.append(json.loads(data))
    return events


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["plane"] == "demo"
    assert body["variants"] >= 1


def test_variants_camelcase_and_precisions(client):
    variants = client.get("/variants").json()
    assert variants, "no variants returned"
    # camelCase aliases the frontend contract requires
    assert all("sizeGB" in v and "vramGB" in v and "stepsPerSec" in v for v in variants)
    precisions = {v["precision"] for v in variants}
    assert "FP8" in precisions and "INT4" not in precisions  # post-TRT-migration matrix


def test_generate_streams_load_denoise_decode_then_done(client):
    body = {"prompt": "a test", "variantId": "fp16-base", "steps": 3,
            "seed": 1, "width": 256, "height": 256}
    with client.stream("POST", "/generate", json=body) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        events = _sse_events(resp)

    stages = {e["stage"] for e in events if e["type"] == "status"}
    assert {"load", "denoise", "decode"} <= stages

    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    result = done[0]["result"]
    assert result["imageUrl"].startswith("data:image/png;base64,")
    assert result["metrics"]["latencyMs"]["total"] > 0          # camelCase metrics
    assert result["metrics"]["throughputStepsPerSec"] > 0

    progress = [e["step"] for e in events if e["type"] == "progress"]
    assert progress[-1] == 3                                     # every step reported


def test_generate_clamps_to_server_guardrails(client, monkeypatch):
    monkeypatch.setattr(main.config, "MAX_STEPS", 2)
    body = {"prompt": "x", "variantId": "fp16-base", "steps": 5,
            "seed": 1, "width": 256, "height": 256}
    with client.stream("POST", "/generate", json=body) as resp:
        events = _sse_events(resp)
    result = next(e["result"] for e in events if e["type"] == "done")
    assert result["params"]["steps"] == 2                       # clamped from 5


def test_unknown_variant_streams_error_not_500(client):
    body = {"prompt": "x", "variantId": "does-not-exist", "steps": 2,
            "seed": 1, "width": 256, "height": 256}
    with client.stream("POST", "/generate", json=body) as resp:
        assert resp.status_code == 200                          # error is in-stream
        events = _sse_events(resp)
    assert any(e["type"] == "error" for e in events)


def test_empty_prompt_is_rejected(client):
    resp = client.post("/generate", json={"prompt": "", "variantId": "fp16-base"})
    assert resp.status_code == 422                              # schema min_length=1
