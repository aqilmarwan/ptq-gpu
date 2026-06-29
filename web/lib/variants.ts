import type { Preset, Variant } from "./types";

/**
 * Mock variant catalog — mirrors `inference/variants.yaml`.
 *
 * Numbers are representative of SDXL 1.0 on a single 24GB GPU and are only used
 * for the standalone demo. When the FastAPI `/variants` endpoint is reachable,
 * its (real, MLflow-registered) values replace these.
 */
export const MOCK_VARIANTS: Variant[] = [
  {
    id: "fp16-base",
    label: "FP16 · Base",
    precision: "FP16",
    style: "Base",
    base: "SDXL 1.0",
    sizeGB: 13.0,
    vramGB: 18.4,
    stepsPerSec: 6.1,
    quality: 98,
    defaultSteps: 30,
    blurb: "Reference quality. The baseline every variant is measured against.",
    licence: "CreativeML OpenRAIL++-M",
  },
  {
    id: "int8-base",
    label: "INT8 · Base",
    precision: "INT8",
    style: "Base",
    base: "SDXL 1.0",
    sizeGB: 7.1,
    vramGB: 11.2,
    stepsPerSec: 9.4,
    quality: 95,
    defaultSteps: 30,
    blurb: "optimum-quanto weight-only INT8. Near-FP16 quality, ~45% lighter.",
    licence: "CreativeML OpenRAIL++-M",
  },
  {
    id: "int4-base",
    label: "INT4 · Base",
    precision: "INT4",
    style: "Base",
    base: "SDXL 1.0",
    sizeGB: 4.3,
    vramGB: 7.6,
    stepsPerSec: 12.8,
    quality: 90,
    defaultSteps: 28,
    blurb: "NF4 double-quant via bitsandbytes. Fits comfortably, fastest base.",
    licence: "CreativeML OpenRAIL++-M",
  },
  {
    id: "fp16-lora",
    label: "FP16 · LoRA",
    precision: "FP16",
    style: "LoRA",
    base: "SDXL 1.0",
    loraName: "Neon Atlas",
    sizeGB: 13.2,
    vramGB: 18.7,
    stepsPerSec: 5.9,
    quality: 97,
    defaultSteps: 30,
    blurb: "Custom DreamBooth-LoRA on FP16 base. Signature look, full fidelity.",
    licence: "OpenRAIL++-M + LoRA (CC-BY)",
  },
  {
    id: "int4-lora",
    label: "INT4 · LoRA",
    precision: "INT4",
    style: "LoRA",
    base: "SDXL 1.0",
    loraName: "Neon Atlas",
    sizeGB: 4.5,
    vramGB: 7.9,
    stepsPerSec: 12.2,
    quality: 88,
    defaultSteps: 28,
    blurb: "The LoRA style, quantised. The cheapest way to ship the brand look.",
    licence: "OpenRAIL++-M + LoRA (CC-BY)",
  },
];

export const MOCK_PRESETS: Preset[] = [
  {
    id: "fast",
    label: "Fast",
    variantId: "int4-base",
    steps: 6,
    guidance: 2.0,
    blurb: "Turbo schedule on INT4 — a frame in a heartbeat.",
  },
  {
    id: "quality",
    label: "Quality",
    variantId: "fp16-base",
    steps: 30,
    guidance: 6.5,
    blurb: "Full step count on FP16 — every detail, no compromise.",
  },
];

export const EXAMPLE_PROMPTS = [
  "a lone lighthouse on a basalt cliff, bioluminescent surf, volumetric fog, cinematic",
  "macro shot of a dew-covered dragonfly wing, iridescent, shallow depth of field",
  "retro-futurist transit poster of a floating city, risograph texture, muted teal and coral",
  "portrait of an arctic fox made of cut paper, studio lighting, intricate layered shadows",
];

export function precisionColor(p: Variant["precision"]): string {
  return p === "FP16" ? "var(--color-fp16)" : p === "INT8" ? "var(--color-int8)" : "var(--color-int4)";
}
