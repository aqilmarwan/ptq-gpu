import { renderArtwork } from "./art";
import type { GenEvent, GenerationParams, GenerationResult, Variant } from "./types";
import { MOCK_VARIANTS } from "./variants";

/**
 * Inference API client.
 *
 * Talks to the FastAPI service at NEXT_PUBLIC_API_URL when it is reachable, and
 * transparently falls back to an in-browser mock so the studio is fully
 * demoable before the backend exists. Swap nothing when FastAPI comes up.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Variants whose weights are "resident in VRAM" this session (mock warm cache). */
const warmed = new Set<string>();

export async function getVariants(): Promise<{ variants: Variant[]; source: "live" | "mock" }> {
  try {
    const res = await fetch(`${API_BASE}/variants`, { signal: AbortSignal.timeout(2500) });
    if (!res.ok) throw new Error(String(res.status));
    const variants = (await res.json()) as Variant[];
    if (!Array.isArray(variants) || variants.length === 0) throw new Error("empty");
    return { variants, source: "live" };
  } catch {
    return { variants: MOCK_VARIANTS, source: "mock" };
  }
}

export interface GenerateHandlers {
  onEvent: (e: GenEvent) => void;
  signal?: AbortSignal;
}

/** Run a generation, streaming progress events. Resolves with the final result. */
export async function generate(
  params: GenerationParams,
  variant: Variant,
  { onEvent, signal }: GenerateHandlers,
): Promise<GenerationResult> {
  try {
    return await streamFromBackend(params, onEvent, signal);
  } catch (err) {
    if (signal?.aborted) throw err;
    // backend unreachable -> mock
    return mockGenerate(params, variant, onEvent, signal);
  }
}

/* ----------------------------------------------------- real backend (SSE) -- */

async function streamFromBackend(
  params: GenerationParams,
  onEvent: (e: GenEvent) => void,
  signal?: AbortSignal,
): Promise<GenerationResult> {
  const res = await fetch(`${API_BASE}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(params),
    signal: signal ?? AbortSignal.timeout(3000),
  });
  if (!res.ok || !res.body) throw new Error(`generate failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: GenerationResult | null = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; data lives on `data:` lines.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const data = frame
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trim())
        .join("");
      if (!data) continue;
      const evt = JSON.parse(data) as GenEvent;
      onEvent(evt);
      if (evt.type === "done") result = evt.result;
      if (evt.type === "error") throw new Error(evt.message);
    }
  }
  if (!result) throw new Error("stream ended without a result");
  return result;
}

/* --------------------------------------------------------------- mock plane -- */

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("aborted", "AbortError"));
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      reject(new DOMException("aborted", "AbortError"));
    });
  });

async function mockGenerate(
  params: GenerationParams,
  variant: Variant,
  onEvent: (e: GenEvent) => void,
  signal?: AbortSignal,
): Promise<GenerationResult> {
  const cold = !warmed.has(variant.id);
  // cold load scales with weight size; a 13GB FP16 load costs more than a 7GB FP8
  const coldLoadMs = cold ? 1400 + variant.sizeGB * 230 : 0;

  onEvent({
    type: "status",
    stage: "load",
    message: cold ? `Cold-loading ${variant.label} weights into VRAM…` : `${variant.label} resident — warm start`,
    cold,
  });
  await sleep(cold ? 520 : 120, signal); // representative slice; full cost is reported in metrics
  warmed.add(variant.id);

  // denoise loop — emit a progress tick per step, paced by the variant throughput
  onEvent({ type: "status", stage: "denoise", message: "Running denoising loop…", cold });
  const perStepMs = 1000 / variant.stepsPerSec;
  // animate faster than real-time so the demo stays snappy on slow variants
  const tick = Math.min(perStepMs, 90);
  const preview = renderArtwork(params.prompt, params.seed, variant);
  for (let step = 1; step <= params.steps; step++) {
    await sleep(tick, signal);
    onEvent({
      type: "progress",
      step,
      totalSteps: params.steps,
      previewUrl: step >= Math.ceil(params.steps * 0.4) ? preview : undefined,
    });
  }

  onEvent({ type: "status", stage: "decode", message: "VAE decoding latents…", cold });
  await sleep(220, signal);

  const denoiseMs = (params.steps / variant.stepsPerSec) * 1000;
  const vaeDecodeMs = 180 + variant.vramGB * 6;
  const total = coldLoadMs + denoiseMs + vaeDecodeMs;

  const result: GenerationResult = {
    imageUrl: renderArtwork(params.prompt, params.seed, variant),
    variantId: variant.id,
    params,
    metrics: {
      cold,
      latencyMs: {
        coldLoad: Math.round(coldLoadMs),
        denoise: Math.round(denoiseMs),
        vaeDecode: Math.round(vaeDecodeMs),
        total: Math.round(total),
      },
      throughputStepsPerSec: variant.stepsPerSec,
      vramPeakGB: Number((variant.vramGB + 0.3).toFixed(1)),
    },
  };
  onEvent({ type: "done", result });
  return result;
}

export { API_BASE };
