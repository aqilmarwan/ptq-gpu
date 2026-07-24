#!/usr/bin/env python3
"""Wait for the endpoint to be ready, call /generate, save the image, print metrics.

Usage:
    python3 scripts/generate.py <base-url> [prompt] [variant]

Example:
    python3 scripts/generate.py http://localhost:8000
    python3 scripts/generate.py http://localhost:8000 "a lighthouse at dusk" int8-base

Stdlib only. Polls /healthz first because a freshly-deployed pod syncs its engines
from S3 on startup (the init container) and isn't ready until that finishes --
polling health sidesteps a hang on the first request during that window.
"""

import base64
import json
import sys
import time
import urllib.error
import urllib.request

base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else sys.exit("usage: generate.py <base-url> [prompt] [variant]")
prompt = sys.argv[2] if len(sys.argv) > 2 else "spongebob with friends"
variant = sys.argv[3] if len(sys.argv) > 3 else "fp16-base"


def wait_for_ready(timeout_s: int = 900) -> None:
    """Poll /healthz until the container is up (engines synced, Registry loaded)."""
    print(f"waiting for {base}/healthz (cold start syncs engines from S3)...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/healthz", timeout=15) as r:
                body = json.loads(r.read())
                print(f"ready · plane={body.get('plane')} · variants={body.get('variants')}")
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            print("  still starting...", flush=True)
            time.sleep(10)
    sys.exit("timed out waiting for the container to become ready")


wait_for_ready()

body = json.dumps({
    "prompt": prompt, "variantId": variant,
    "steps": 20, "seed": 1, "width": 1024, "height": 1024,
}).encode()
req = urllib.request.Request(base + "/generate", data=body, headers={"Content-Type": "application/json"})

result = None
print(f"POST /generate  variant={variant!r}  prompt={prompt!r}")
with urllib.request.urlopen(req, timeout=300) as resp:      # first gen deserializes the engine (slow)
    for raw in resp:
        line = raw.decode("utf-8")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        evt = json.loads(data)
        kind = evt.get("type")
        if kind == "status":
            print(f"  {evt['stage']}: {evt['message']}")
        elif kind == "progress":
            print(f"\r  step {evt['step']}/{evt['totalSteps']}", end="", flush=True)
        elif kind == "error":
            sys.exit(f"\nerror from server: {evt['message']}")
        elif kind == "done":
            result = evt["result"]

if not result:
    sys.exit("\nstream ended without a result")

png = base64.b64decode(result["imageUrl"].split(",", 1)[1])
with open("out.png", "wb") as f:
    f.write(png)
print(f"\nwrote out.png ({len(png) // 1024} KB)")
print("metrics:", json.dumps(result["metrics"], indent=2))
