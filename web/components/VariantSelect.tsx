"use client";

import { useEffect, useRef, useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react";
import type { Variant } from "@/lib/types";
import { PrecisionTag } from "./PrecisionTag";
import { cn } from "@/lib/utils";

/** Compact rich dropdown for picking a variant in a Compare slot. */
export function VariantSelect({
  variants,
  value,
  onChange,
  accent,
  disabled,
}: {
  variants: Variant[];
  value: string;
  onChange: (id: string) => void;
  accent: string;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const current = variants.find((v) => v.id === value) ?? variants[0];

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 rounded-xl border p-2.5 text-left transition-colors disabled:opacity-50"
        style={{ borderColor: open ? accent : "var(--color-line)" }}
      >
        <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: accent }} />
        <PrecisionTag precision={current.precision} size="xs" />
        <span className="text-[13px] font-medium">{current.style === "LoRA" ? current.loraName : "Base"}</span>
        <span className="tabular ml-auto text-[11px] text-faint">{current.stepsPerSec.toFixed(1)} it/s</span>
        <ChevronsUpDown className="h-3.5 w-3.5 text-faint" />
      </button>

      {open && (
        <div className="absolute z-30 mt-1.5 w-full overflow-hidden rounded-xl border border-line-strong bg-panel-2 p-1 shadow-2xl shadow-black/50">
          {variants.map((v) => (
            <button
              key={v.id}
              type="button"
              onClick={() => {
                onChange(v.id);
                setOpen(false);
              }}
              className={cn(
                "flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left transition-colors hover:bg-white/5",
                v.id === value && "bg-white/[0.04]",
              )}
            >
              <PrecisionTag precision={v.precision} size="xs" />
              <span className="text-[12px] font-medium">{v.style === "LoRA" ? v.loraName : "Base"}</span>
              <span className="tabular ml-auto text-[11px] text-faint">{v.sizeGB.toFixed(1)}G · q{v.quality}</span>
              {v.id === value && <Check className="h-3.5 w-3.5" style={{ color: accent }} />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
