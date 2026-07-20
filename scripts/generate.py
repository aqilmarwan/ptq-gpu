#!/usr/bin/env python3
"""Call the /generate SSE endpoint, save the image to out.png, print metrics.

Usage:
    python3 scripts/generate.py <base-url> [prompt] [variant]

Example:
    python3 scripts/generate.py https://mrwndevs--ptq-gpu-build-serve.modal.run
    python3 scripts/generate.py https://...modal.run "a lighthouse at dusk" int8-base

Stdlib only -- no pip installs. Avoids the shell-quoting/curl-multiline mess.
"""

import base64
import json
import sys
import urllib.request

base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else sys.exit("usage: generate.py <base-url> [prompt] [variant]")
prompt = sys.argv[2] if len(sys.argv) > 2 else "a red fox in snow, sharp focus, natural light"
variant = sys.argv[3] if len(sys.argv) > 3 else "fp16-base"

body = json.dumps({
    "prompt": prompt, "variantId": variant,
    "steps": 20, "seed": 1, "width": 1024, "height": 1024,
}).encode()

req = urllib.request.Request(
    base + "/generate", data=body, headers={"Content-Type": "application/json"}
)

result = None
print(f"POST {base}/generate  variant={variant!r}  prompt={prompt!r}")
with urllib.request.urlopen(req) as resp:
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
