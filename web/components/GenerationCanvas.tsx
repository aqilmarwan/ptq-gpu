"use client";

import { AnimatePresence, motion } from "motion/react";
import { ImageIcon, Loader2 } from "lucide-react";
import type { GenState } from "@/lib/useGeneration";
import type { Variant } from "@/lib/types";
import { PrecisionTag } from "./PrecisionTag";
import { cn } from "@/lib/utils";

const STAGE_LABEL: Record<string, string> = {
  load: "Loading weights",
  denoise: "Denoising",
  decode: "Decoding",
  done: "Done",
};

export function GenerationCanvas({
  state,
  variant,
  className,
  rounded = "rounded-3xl",
}: {
  state: GenState;
  variant: Variant;
  className?: string;
  rounded?: string;
}) {
  const { stage, step, totalSteps, previewUrl, result } = state;
  const generating = state.running;
  const img = result?.imageUrl ?? previewUrl;
  const progress = totalSteps > 0 ? step / totalSteps : 0;

  return (
    <div
      className={cn(
        "relative aspect-square w-full overflow-hidden border border-line bg-ink-2",
        rounded,
        className,
      )}
    >
      {/* image / preview layer */}
      <AnimatePresence mode="wait">
        {img ? (
          <motion.img
            key={img.slice(0, 48)}
            src={img}
            alt={result ? "Generated image" : "Live preview"}
            initial={{ opacity: 0, scale: 1.02 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className={cn(
              "absolute inset-0 h-full w-full object-cover transition-[filter] duration-500",
              generating && !result ? "blur-md brightness-90" : "blur-0",
            )}
          />
        ) : (
          <EmptyState key="empty" />
        )}
      </AnimatePresence>

      {/* streaming overlay */}
      <AnimatePresence>
        {generating && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 grid place-items-center bg-ink/40 backdrop-blur-[2px]"
          >
            <ProgressRing progress={progress} stage={stage} step={step} total={totalSteps} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* variant tag (when an image is present) */}
      {img && (
        <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-1.5">
          <span className="flex items-center gap-1.5 rounded-lg bg-ink/70 px-2 py-1 text-[11px] font-medium backdrop-blur">
            <PrecisionTag precision={variant.precision} size="xs" />
            <span className="text-muted">{variant.style === "LoRA" ? variant.loraName : "Base"}</span>
          </span>
        </div>
      )}

      {/* bottom progress bar while denoising */}
      {generating && (
        <div className="absolute inset-x-0 bottom-0 h-1 bg-white/5">
          <motion.div
            className="h-full"
            style={{ background: "linear-gradient(90deg, var(--color-accent-2), var(--color-accent))" }}
            animate={{ width: `${Math.round(progress * 100)}%` }}
            transition={{ ease: "linear", duration: 0.1 }}
          />
        </div>
      )}
    </div>
  );
}

function ProgressRing({
  progress,
  stage,
  step,
  total,
}: {
  progress: number;
  stage: string;
  step: number;
  total: number;
}) {
  const R = 34;
  const C = 2 * Math.PI * R;
  const denoising = stage === "denoise";
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative h-24 w-24">
        <svg viewBox="0 0 80 80" className="h-full w-full -rotate-90">
          <circle cx="40" cy="40" r={R} fill="none" stroke="var(--color-line-strong)" strokeWidth="4" />
          <circle
            cx="40"
            cy="40"
            r={R}
            fill="none"
            stroke="url(#ring)"
            strokeWidth="4"
            strokeLinecap="round"
            strokeDasharray={C}
            strokeDashoffset={denoising ? C * (1 - progress) : C * 0.75}
            className={denoising ? "transition-[stroke-dashoffset] duration-150" : "animate-spin-slow"}
          />
          <defs>
            <linearGradient id="ring" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="var(--color-accent-2)" />
              <stop offset="100%" stopColor="var(--color-accent)" />
            </linearGradient>
          </defs>
        </svg>
        <div className="absolute inset-0 grid place-items-center">
          {denoising ? (
            <div className="text-center">
              <div className="tabular text-lg font-semibold leading-none">{step}</div>
              <div className="tabular text-[10px] text-faint">/ {total}</div>
            </div>
          ) : (
            <Loader2 className="h-5 w-5 animate-spin text-accent" />
          )}
        </div>
      </div>
      <span className="rounded-full bg-ink/70 px-3 py-1 text-[12px] font-medium text-fg backdrop-blur">
        {STAGE_LABEL[stage] ?? "Working"}
        {denoising ? <span className="tabular text-faint"> · {Math.round(progress * 100)}%</span> : "…"}
      </span>
    </div>
  );
}

function EmptyState() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="absolute inset-0 grid place-items-center"
    >
      <div className="flex flex-col items-center gap-3 text-center">
        <div className="animate-float grid h-16 w-16 place-items-center rounded-2xl border border-line bg-ink-2/60">
          <ImageIcon className="h-7 w-7 text-faint" strokeWidth={1.5} />
        </div>
      </div>
    </motion.div>
  );
}
