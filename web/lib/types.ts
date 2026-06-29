/**
 * Shared types for the Image Gen Studio frontend.
 *
 * These mirror the FastAPI inference contract:
 *   GET  /variants          -> Variant[]
 *   POST /generate (SSE)     -> stream of GenEvent, terminating in { type: "done" }
 *
 * The frontend only ever calls those two endpoints. Model code never ships here.
 */

export type Precision = "FP16" | "INT8" | "INT4";
export type Style = "Base" | "LoRA";

/** A registered, servable model variant (precision × style), as defined in
 *  `inference/variants.yaml` and benchmarked into the MLflow registry. */
export interface Variant {
  id: string; // e.g. "int4-base"
  label: string; // e.g. "INT4 · Base"
  precision: Precision;
  style: Style;
  base: string; // "SDXL 1.0"
  /** human style name when style === "LoRA" */
  loraName?: string;
  /** on-disk weight size in GB */
  sizeGB: number;
  /** expected peak VRAM at inference, GB (from registry benchmark) */
  vramGB: number;
  /** denoising throughput, steps/sec (from registry benchmark) */
  stepsPerSec: number;
  /** quality score 0–100 (CLIP / aesthetic, from registry benchmark) */
  quality: number;
  /** recommended default step count */
  defaultSteps: number;
  /** short one-liner shown in the picker */
  blurb: string;
  /** licence string, surfaced in UI */
  licence: string;
}

export interface Preset {
  id: "fast" | "quality";
  label: string;
  variantId: string;
  steps: number;
  guidance: number;
  blurb: string;
}

export interface GenerationParams {
  prompt: string;
  negativePrompt: string;
  variantId: string;
  steps: number;
  guidance: number;
  seed: number;
  width: number;
  height: number;
}

/** Server-side measured metrics, taken around the real denoising loop. */
export interface Metrics {
  /** true when the variant had to be loaded into VRAM for this request */
  cold: boolean;
  latencyMs: {
    coldLoad: number; // weight load (0 when warm)
    denoise: number; // the UNet loop
    vaeDecode: number; // latent -> pixels
    total: number;
  };
  throughputStepsPerSec: number;
  vramPeakGB: number;
}

export interface GenerationResult {
  imageUrl: string;
  variantId: string;
  params: GenerationParams;
  metrics: Metrics;
}

/** Server-Sent Events emitted by POST /generate. */
export type GenEvent =
  | { type: "status"; stage: "load" | "denoise" | "decode"; message: string; cold: boolean }
  | { type: "progress"; step: number; totalSteps: number; previewUrl?: string }
  | { type: "done"; result: GenerationResult }
  | { type: "error"; message: string };

export interface ApiHealth {
  /** "live" when the FastAPI service answered, "mock" when we fell back */
  source: "live" | "mock";
}
