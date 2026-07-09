"use client";

import { Boxes, HardDrive, Sparkle, Zap } from "lucide-react";
import type { Precision, Style, Variant } from "@/lib/types";
import { precisionColor } from "@/lib/variants";
import { cn } from "@/lib/utils";

const PRECISIONS: Precision[] = ["FP16", "INT8", "FP8"];

/** Resolve the variant for a (style, precision) cell, tolerating sparse matrices. */
function resolve(variants: Variant[], style: Style, precision: Precision): Variant | undefined {
  return (
    variants.find((v) => v.style === style && v.precision === precision) ??
    variants.find((v) => v.style === style)
  );
}

export function VariantPicker({
  variants,
  value,
  onChange,
  disabled,
}: {
  variants: Variant[];
  value: string;
  onChange: (variantId: string) => void;
  disabled?: boolean;
}) {
  const current = variants.find((v) => v.id === value) ?? variants[0];
  const styles = Array.from(new Set(variants.map((v) => v.style))) as Style[];
  const loraName = variants.find((v) => v.style === "LoRA")?.loraName ?? "LoRA";

  const pick = (style: Style, precision: Precision) => {
    const v = resolve(variants, style, precision);
    if (v) onChange(v.id);
  };

  return (
    <div className="space-y-3">
      {/* style axis */}
      <Segmented
        disabled={disabled}
        options={styles.map((s) => ({
          key: s,
          label: s === "LoRA" ? loraName : "Base",
          icon: s === "LoRA" ? Sparkle : Boxes,
          active: current.style === s,
          onClick: () => pick(s, current.precision),
        }))}
      />

      {/* precision axis */}
      <div className="grid grid-cols-3 gap-2">
        {PRECISIONS.map((p) => {
          const target = resolve(variants, current.style, p);
          const exists = target?.precision === p;
          const active = current.precision === p && exists;
          const color = precisionColor(p);
          return (
            <button
              key={p}
              type="button"
              disabled={disabled || !exists}
              onClick={() => pick(current.style, p)}
              className={cn(
                "group relative rounded-xl border p-2.5 text-left transition-all disabled:cursor-not-allowed disabled:opacity-30",
                active ? "border-transparent" : "border-line hover:border-line-strong",
              )}
              style={active ? { boxShadow: `inset 0 0 0 1.5px ${color}`, background: `color-mix(in oklab, ${color} 10%, transparent)` } : undefined}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-[13px] font-semibold" style={{ color: active ? color : undefined }}>
                  {p}
                </span>
                <span className="h-2 w-2 rounded-full" style={{ background: color, opacity: active ? 1 : 0.35 }} />
              </div>
              <div className="tabular mt-1 text-[10px] text-faint">
                {target ? `${target.stepsPerSec.toFixed(1)} it/s` : "—"}
              </div>
            </button>
          );
        })}
      </div>

      {/* resolved variant readout */}
      <VariantReadout variant={current} />
    </div>
  );
}

function VariantReadout({ variant }: { variant: Variant }) {
  return (
    <div className="rounded-xl border border-line bg-ink-2/50 p-3">
      <p className="mb-2.5 text-[12px] leading-relaxed text-muted">{variant.blurb}</p>
      <div className="grid grid-cols-3 gap-2">
        <Mini icon={HardDrive} label="size" value={`${variant.sizeGB.toFixed(1)}G`} />
        <Mini icon={Zap} label="VRAM" value={`${variant.vramGB.toFixed(1)}G`} />
        <Mini icon={Sparkle} label="quality" value={`${variant.quality}`} />
      </div>
      <div className="mt-3 space-y-1.5">
        <QualityBar quality={variant.quality} />
      </div>
    </div>
  );
}

function Mini({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
}) {
  return (
    <div>
      <div className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-faint">
        <Icon className="h-3 w-3" />
        {label}
      </div>
      <div className="tabular text-[13px] font-semibold text-fg">{value}</div>
    </div>
  );
}

function QualityBar({ quality }: { quality: number }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px] text-faint">
        <span>quality vs FP16 reference</span>
        <span className="tabular">{quality}/100</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-ink ring-1 ring-line">
        <div
          className="h-full rounded-full"
          style={{
            width: `${quality}%`,
            background: "linear-gradient(90deg, var(--color-fp8), var(--color-int8), var(--color-fp16))",
          }}
        />
      </div>
    </div>
  );
}

function Segmented({
  options,
  disabled,
}: {
  disabled?: boolean;
  options: {
    key: string;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    active: boolean;
    onClick: () => void;
  }[];
}) {
  return (
    <div className="grid grid-cols-2 gap-1 rounded-xl border border-line bg-ink-2/60 p-1">
      {options.map((o) => (
        <button
          key={o.key}
          type="button"
          disabled={disabled}
          onClick={o.onClick}
          className={cn(
            "flex items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-[12px] font-medium transition-colors disabled:opacity-50",
            o.active ? "bg-white/10 text-fg" : "text-muted hover:text-fg",
          )}
        >
          <o.icon className="h-3.5 w-3.5" />
          {o.label}
        </button>
      ))}
    </div>
  );
}
