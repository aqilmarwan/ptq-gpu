"""API request/response models.

Field names are emitted in **camelCase** to match the frontend contract in
`web/lib/types.ts` exactly. A camelCase alias generator handles most fields; the
``*GB`` fields get explicit aliases because the frontend uppercases "GB".
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

Precision = Literal["FP16", "INT8", "FP8"]
Style = Literal["Base", "LoRA"]


class _Camel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Variant(_Camel):
    """One entry of GET /variants -- display + benchmarked registry metrics."""

    id: str
    label: str
    precision: Precision
    style: Style
    base: str
    lora_name: Optional[str] = None
    size_gb: float = Field(alias="sizeGB")
    vram_gb: float = Field(alias="vramGB")
    steps_per_sec: float
    quality: int
    default_steps: int
    blurb: str
    licence: str


class GenerationParams(_Camel):
    """POST /generate body. Also echoed back inside the result."""

    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    variant_id: str
    steps: int = Field(default=30, ge=1, le=60)
    guidance: float = Field(default=6.5, ge=0, le=20)
    seed: int = 0
    width: int = Field(default=1024, ge=256, le=1024)
    height: int = Field(default=1024, ge=256, le=1024)


class LatencyBreakdown(_Camel):
    cold_load: int
    denoise: int
    vae_decode: int
    total: int


class Metrics(_Camel):
    cold: bool
    latency_ms: LatencyBreakdown
    throughput_steps_per_sec: float
    vram_peak_gb: float = Field(alias="vramPeakGB")


class GenerationResult(_Camel):
    image_url: str
    variant_id: str
    params: GenerationParams
    metrics: Metrics
