"""FastAPI inference service for Quant Studio.

Two endpoints, matching the frontend contract in `web/lib/types.ts`:

    GET  /variants   -> Variant[]            (the precision x style matrix)
    POST /generate   -> text/event-stream    (status + progress + done events)

Generation is GPU-bound and synchronous, so each request runs the blocking work
in a thread and bridges its progress callbacks onto an asyncio queue that is
drained as Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import config
from pipelines import Registry
from schemas import GenerationParams

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("studio.api")

_SENTINEL = object()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = Registry()
    log.info("Registry ready on the %s plane", app.state.registry.plane)
    yield


app = FastAPI(title="Quant Studio Inference", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _registry(request: Request) -> Registry:
    return request.app.state.registry


@app.get("/")
def root(request: Request):
    reg = _registry(request)
    return {
        "service": "quant-studio-inference",
        "plane": reg.plane,
        "variants": len(reg.variants()),
        "docs": "/docs",
    }


@app.get("/healthz")
def healthz(request: Request):
    return {"status": "ok", "plane": _registry(request).plane}


@app.get("/variants")
def variants(request: Request):
    return [v.model_dump(by_alias=True) for v in _registry(request).variants()]


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/generate")
async def generate(params: GenerationParams, request: Request):
    reg = _registry(request)
    if not reg.has(params.variant_id):
        raise HTTPException(status_code=404, detail=f"unknown variant: {params.variant_id}")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(event) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def work() -> None:
        try:
            result = reg.run(params, emit)
            emit({"type": "done", "result": result.model_dump(by_alias=True)})
        except Exception as exc: 
            log.exception("generation failed")
            emit({"type": "error", "message": str(exc)})
        finally:
            emit(_SENTINEL)

    async def stream():
        fut = loop.run_in_executor(None, work)
        try:
            while True:
                event = await queue.get()
                if event is _SENTINEL:
                    break
                if await request.is_disconnected():
                    break
                yield _sse(event)
        finally:
            await fut

        
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # disable proxy buffering so SSE flushes live
    }
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)
