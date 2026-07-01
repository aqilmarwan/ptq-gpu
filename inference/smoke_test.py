"""In-process contract smoke test (no network, no GPU).

Boots the FastAPI app on the demo plane via httpx's ASGI transport and asserts
the /variants + /generate (SSE) contract the frontend depends on. Run in CI:

    STUDIO_DEMO=1 python smoke_test.py
"""

from __future__ import annotations

import asyncio
import json
import os

os.environ.setdefault("STUDIO_DEMO", "1")

import httpx  # noqa: E402

import main  # noqa: E402
from pipelines import Registry  # noqa: E402


async def main_async() -> None:
    # initialise the registry directly (skip the ASGI lifespan for portability)
    main.app.state.registry = Registry()
    assert main.app.state.registry.plane == "demo"

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=60) as c:
        # ---- GET /variants --------------------------------------------------
        r = await c.get("/variants")
        assert r.status_code == 200, r.status_code
        variants = r.json()
        assert len(variants) == 5, f"expected 5 variants, got {len(variants)}"
        need = {"id", "label", "precision", "style", "base", "sizeGB", "vramGB",
                "stepsPerSec", "quality", "defaultSteps", "blurb", "licence"}
        missing = need - set(variants[0])
        assert not missing, f"variant missing keys: {missing}"
        assert any(v.get("loraName") for v in variants), "no LoRA variant present"

        # ---- POST /generate (SSE) ------------------------------------------
        body = {"prompt": "ci smoke", "negativePrompt": "blurry",
                "variantId": "int4-base", "steps": 4, "guidance": 6.0,
                "seed": 7, "width": 1024, "height": 1024}
        events: list[dict] = []
        async with c.stream("POST", "/generate", json=body) as resp:
            assert resp.status_code == 200, resp.status_code
            assert "text/event-stream" in resp.headers["content-type"]
            buf = ""
            async for chunk in resp.aiter_text():
                buf += chunk
                while "\n\n" in buf:
                    frame, buf = buf.split("\n\n", 1)
                    data = "".join(l[5:].strip() for l in frame.splitlines() if l.startswith("data:"))
                    if data:
                        events.append(json.loads(data))

        types = [e["type"] for e in events]
        assert types[0] == "status", types[:1]
        assert types.count("progress") == 4, f"expected 4 progress, got {types.count('progress')}"
        assert types[-1] == "done", types[-1]
        result = events[-1]["result"]
        assert result["imageUrl"].startswith("data:image/png;base64,")
        assert result["variantId"] == "int4-base"
        assert result["params"]["negativePrompt"] == "blurry"
        m = result["metrics"]
        assert set(m["latencyMs"]) == {"coldLoad", "denoise", "vaeDecode", "total"}
        assert {"throughputStepsPerSec", "vramPeakGB", "cold"} <= set(m)

        # ---- 404 on unknown variant ----------------------------------------
        bad = await c.post("/generate", json={**body, "variantId": "nope"})
        assert bad.status_code == 404, bad.status_code

    print(f"smoke OK: {len(variants)} variants, {len(events)} stream events, contract matches")


if __name__ == "__main__":
    asyncio.run(main_async())
