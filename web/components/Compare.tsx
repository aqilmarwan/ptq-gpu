"use client";

import { useEffect, useMemo, useState } from "react";
import { Dices, Lock, Sparkles, Unlock } from "lucide-react";
import { getVariants } from "@/lib/api";
import { MOCK_VARIANTS } from "@/lib/variants";
import type { GenerationParams, GenerationResult, Variant } from "@/lib/types";
import { useGeneration } from "@/lib/useGeneration";
import { cn, fmtGB, fmtMs, randomSeed } from "@/lib/utils";
import { readHandoff } from "@/lib/handoff";
import { GenerationCanvas } from "./GenerationCanvas";
import { MetricsPanel } from "./MetricsPanel";
import { VariantSelect } from "./VariantSelect";
import { Slider } from "./Slider";

const ACCENT_A = "var(--color-accent)";
const ACCENT_B = "var(--color-fp16)";

export function Compare() {
  const [variants, setVariants] = useState<Variant[]>(MOCK_VARIANTS);
  const [prompt, setPrompt] = useState(
    "a lone lighthouse on a basalt cliff, bioluminescent surf, volumetric fog, cinematic",
  );
  const [seed, setSeed] = useState(742183);
  const [lockSeed, setLockSeed] = useState(true);
  const [steps, setSteps] = useState(28);
  const [guidance, setGuidance] = useState(6.0);
  const [aId, setAId] = useState("fp16-base");
  const [bId, setBId] = useState("int4-base");

  const a = useGeneration();
  const b = useGeneration();

  const varA = useMemo(() => variants.find((v) => v.id === aId) ?? variants[0], [variants, aId]);
  const varB = useMemo(() => variants.find((v) => v.id === bId) ?? variants[0], [variants, bId]);
  const running = a.state.running || b.state.running;

  useEffect(() => {
    let cancelled = false;
    // Resolve variants and any Studio handoff together, inside the async
    // callback — keeps state updates out of the synchronous effect body.
    getVariants().then(({ variants }) => {
      if (cancelled) return;
      const h = readHandoff();
      const has = (id: string) => variants.some((v) => v.id === id);
      setVariants(variants);
      setAId((id) => (h && has(h.variantId) ? h.variantId : has(id) ? id : variants[0].id));
      setBId((id) => (has(id) ? id : variants[variants.length - 1].id));
      if (h) {
        setPrompt(h.prompt);
        setSeed(h.seed);
        setSteps(h.steps);
        setGuidance(h.guidance);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const generateBoth = () => {
    if (!prompt.trim() || running) return;
    const s = lockSeed ? seed : randomSeed();
    if (!lockSeed) setSeed(s);
    const base = { prompt: prompt.trim(), negativePrompt: "", steps, guidance, seed: s, width: 1024, height: 1024 };
    a.run({ ...base, variantId: varA.id } as GenerationParams, varA);
    b.run({ ...base, variantId: varB.id } as GenerationParams, varB);
  };

  const bothDone = a.state.result && b.state.result;

  return (
    <div className="mx-auto max-w-[1400px] space-y-5 px-5 py-6">
      {/* header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-[22px] font-semibold tracking-tight">Compare variants</h1>
        <p className="text-[13px] text-muted">
          One prompt, one seed, two variants — so the speed↔quality tradeoff is the only variable.
        </p>
      </div>

      {/* shared controls */}
      <div className="card grid gap-4 p-4 lg:grid-cols-[1fr_auto]">
        <div className="space-y-3">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={2}
            placeholder="Shared prompt for both variants…"
            className="w-full resize-none rounded-xl border border-line bg-ink-2/60 p-3 text-[13px] leading-relaxed text-fg outline-none focus:border-line-strong"
          />
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-2">
            <Slider label="Steps" value={steps} min={1} max={50} suffix=" steps" disabled={running} onChange={setSteps} />
            <Slider label="Guidance" value={guidance} min={0} max={15} step={0.5} disabled={running} onChange={setGuidance} />
          </div>
        </div>

        <div className="flex flex-col justify-between gap-3 lg:w-[230px]">
          <div className="space-y-2">
            <label className="text-[12px] font-medium text-muted">Shared seed</label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                value={seed}
                disabled={running}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="tabular h-9 w-full rounded-lg border border-line bg-ink-2/60 px-3 text-[13px] text-fg outline-none focus:border-line-strong disabled:opacity-50"
              />
              <button
                type="button"
                title={lockSeed ? "Seed locked" : "Seed roams"}
                onClick={() => setLockSeed((l) => !l)}
                className={cn(
                  "grid h-9 w-9 shrink-0 place-items-center rounded-lg border transition-colors",
                  lockSeed ? "border-accent/50 bg-accent/10 text-accent" : "border-line text-muted hover:text-fg",
                )}
              >
                {lockSeed ? <Lock className="h-3.5 w-3.5" /> : <Unlock className="h-3.5 w-3.5" />}
              </button>
              <button
                type="button"
                title="Randomize"
                onClick={() => setSeed(randomSeed())}
                disabled={running}
                className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-line text-muted transition-colors hover:text-fg disabled:opacity-40"
              >
                <Dices className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
          <button
            type="button"
            onClick={generateBoth}
            disabled={running || !prompt.trim()}
            className="btn-glow flex h-11 items-center justify-center gap-2 rounded-xl text-[14px] font-semibold text-ink disabled:opacity-50"
          >
            <Sparkles className="h-4 w-4" strokeWidth={2.5} />
            {running ? "Rendering…" : "Generate both"}
          </button>
        </div>
      </div>

      {/* head to head */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Slot accent={ACCENT_A} label="A" variants={variants} variantId={aId} onPick={setAId} gen={a} disabled={running} />
        <Slot accent={ACCENT_B} label="B" variants={variants} variantId={bId} onPick={setBId} gen={b} disabled={running} />
      </div>

      {/* verdict */}
      {bothDone && (
        <DiffCard varA={varA} varB={varB} resA={a.state.result!} resB={b.state.result!} />
      )}
    </div>
  );
}

function Slot({
  accent,
  label,
  variants,
  variantId,
  onPick,
  gen,
  disabled,
}: {
  accent: string;
  label: string;
  variants: Variant[];
  variantId: string;
  onPick: (id: string) => void;
  gen: ReturnType<typeof useGeneration>;
  disabled?: boolean;
}) {
  const variant = variants.find((v) => v.id === variantId) ?? variants[0];
  return (
    <div className="card space-y-3 p-4">
      <div className="flex items-center gap-2">
        <span className="grid h-6 w-6 place-items-center rounded-md text-[12px] font-bold text-ink" style={{ background: accent }}>
          {label}
        </span>
        <div className="flex-1">
          <VariantSelect variants={variants} value={variantId} onChange={onPick} accent={accent} disabled={disabled} />
        </div>
      </div>
      <GenerationCanvas state={gen.state} variant={variant} rounded="rounded-2xl" />
      {gen.state.result ? (
        <MetricsPanel metrics={gen.state.result.metrics} compact />
      ) : (
        <div className="tabular flex items-center justify-between rounded-xl border border-line bg-ink-2/40 px-3 py-2.5 text-[11px] text-faint">
          <span>expected</span>
          <span>{((28 / variant.stepsPerSec) * 1000 / 1000).toFixed(1)}s · {variant.vramGB.toFixed(1)}G · q{variant.quality}</span>
        </div>
      )}
    </div>
  );
}

function DiffCard({
  varA,
  varB,
  resA,
  resB,
}: {
  varA: Variant;
  varB: Variant;
  resA: GenerationResult;
  resB: GenerationResult;
}) {
  const speedup = resA.metrics.latencyMs.total / resB.metrics.latencyMs.total; // >1 => B faster
  const fasterB = speedup > 1;
  const ratio = fasterB ? speedup : 1 / speedup;
  const lighter = varA.sizeGB - varB.sizeGB; // >0 => B lighter
  const qDelta = varB.quality - varA.quality; // <0 => B lower quality

  const winner = fasterB ? varB : varA;
  const winLabel = fasterB ? "B" : "A";
  const dq = Math.abs(qDelta);

  const headline = `${winner.label} renders this seed ${ratio.toFixed(1)}× faster${
    Math.abs(lighter) > 0.1 ? ` and ${fmtGB(Math.abs(lighter))} ${fasterB ? "lighter" : "heavier"}` : ""
  }${dq > 0 ? `, for ${qDelta < 0 ? "−" : "+"}${dq} quality` : " at matched quality"}.`;

  return (
    <div className="card space-y-4 p-5">
      <div className="flex items-center gap-2">
        <span className="rounded-md bg-good/15 px-2 py-0.5 text-[11px] font-semibold text-good">verdict</span>
        <p className="text-[14px] font-medium">
          Variant <span className="text-grad font-semibold">{winLabel}</span> wins on speed — {headline}
        </p>
      </div>
      <div className="grid gap-3 sm:grid-cols-4">
        <DiffMetric label="latency" a={fmtMs(resA.metrics.latencyMs.total)} b={fmtMs(resB.metrics.latencyMs.total)} winner={resA.metrics.latencyMs.total <= resB.metrics.latencyMs.total ? "A" : "B"} />
        <DiffMetric label="peak VRAM" a={fmtGB(resA.metrics.vramPeakGB)} b={fmtGB(resB.metrics.vramPeakGB)} winner={resA.metrics.vramPeakGB <= resB.metrics.vramPeakGB ? "A" : "B"} />
        <DiffMetric label="model size" a={fmtGB(varA.sizeGB)} b={fmtGB(varB.sizeGB)} winner={varA.sizeGB <= varB.sizeGB ? "A" : "B"} />
        <DiffMetric label="quality" a={`${varA.quality}`} b={`${varB.quality}`} winner={varA.quality >= varB.quality ? "A" : "B"} higherBetter />
      </div>
    </div>
  );
}

function DiffMetric({
  label,
  a,
  b,
  winner,
}: {
  label: string;
  a: string;
  b: string;
  winner: "A" | "B";
  higherBetter?: boolean;
}) {
  return (
    <div className="rounded-xl border border-line bg-ink-2/40 p-3">
      <div className="mb-2 text-[10px] uppercase tracking-wide text-faint">{label}</div>
      <div className="flex items-center justify-between">
        <Side tag="A" value={a} win={winner === "A"} accent={ACCENT_A} />
        <span className="text-[11px] text-faint">vs</span>
        <Side tag="B" value={b} win={winner === "B"} accent={ACCENT_B} align="right" />
      </div>
    </div>
  );
}

function Side({
  tag,
  value,
  win,
  accent,
  align = "left",
}: {
  tag: string;
  value: string;
  win: boolean;
  accent: string;
  align?: "left" | "right";
}) {
  return (
    <div className={cn("flex flex-col", align === "right" && "items-end")}>
      <span className="flex items-center gap-1 text-[10px]" style={{ color: accent }}>
        {tag}
      </span>
      <span className={cn("tabular text-[14px]", win ? "font-semibold text-fg" : "text-muted")}>
        {value}
        {win && <span className="ml-1 text-[10px]" style={{ color: accent }}>●</span>}
      </span>
    </div>
  );
}
