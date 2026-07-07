"use client";

import { clamp } from "@/lib/utils";

/** Compact labelled slider with a dark-green accent fill. */
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  suffix,
  onChange,
  disabled,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  suffix?: string;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  const pctFill = ((clamp(value, min, max) - min) / (max - min)) * 100;
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <label className="text-[12px] font-medium text-muted">{label}</label>
        <span className="tabular text-[12px] text-fg">
          {value}
          {suffix ? <span className="text-faint">{suffix}</span> : null}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="slider-input h-1.5 w-full cursor-pointer appearance-none rounded-full disabled:opacity-40"
        style={{
          background: `linear-gradient(to right, var(--color-accent-2), var(--color-accent) ${pctFill}%, var(--color-line-strong) ${pctFill}%)`,
        }}
      />
      <style jsx>{`
        .slider-input::-webkit-slider-thumb {
          -webkit-appearance: none;
          height: 14px;
          width: 14px;
          border-radius: 999px;
          background: #fff;
          box-shadow: 0 0 0 3px color-mix(in oklab, var(--color-accent) 45%, transparent);
          transition: box-shadow 0.15s;
        }
        .slider-input::-webkit-slider-thumb:hover {
          box-shadow: 0 0 0 5px color-mix(in oklab, var(--color-accent) 55%, transparent);
        }
        .slider-input::-moz-range-thumb {
          height: 14px;
          width: 14px;
          border: none;
          border-radius: 999px;
          background: #fff;
          box-shadow: 0 0 0 3px color-mix(in oklab, var(--color-accent) 45%, transparent);
        }
      `}</style>
    </div>
  );
}
