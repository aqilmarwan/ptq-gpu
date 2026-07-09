import { Snowflake, Flame, Gauge, MemoryStick, Timer } from "lucide-react";
import type { Metrics } from "@/lib/types";
import { cn, fmtGB, fmtMs } from "@/lib/utils";

const SEGMENTS = [
  { key: "coldLoad", label: "cold load", color: "var(--color-fp16)" },
  { key: "denoise", label: "denoise", color: "var(--color-int8)" },
  { key: "vaeDecode", label: "VAE decode", color: "var(--color-fp8)" },
] as const;

/** The payoff: honest, server-side metrics around the real denoising loop. */
export function MetricsPanel({ metrics, compact = false }: { metrics: Metrics; compact?: boolean }) {
  const { latencyMs } = metrics;
  const total = Math.max(1, latencyMs.total);

  return (
    <div className="space-y-4">
      {/* headline numbers */}
      <div className="grid grid-cols-3 gap-2">
        <Stat
          icon={Timer}
          label="latency"
          value={fmtMs(latencyMs.total)}
          tint="var(--color-accent)"
        />
        <Stat
          icon={Gauge}
          label="throughput"
          value={`${metrics.throughputStepsPerSec.toFixed(1)}`}
          unit="it/s"
          tint="var(--color-int8)"
        />
        <Stat
          icon={MemoryStick}
          label="peak VRAM"
          value={fmtGB(metrics.vramPeakGB)}
          tint="var(--color-fp8)"
        />
      </div>

      {/* latency breakdown */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-medium uppercase tracking-wide text-faint">
            latency breakdown
          </span>
          <ColdWarmBadge cold={metrics.cold} />
        </div>
        <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-ink-2 ring-1 ring-line">
          {SEGMENTS.map((s) => {
            const v = latencyMs[s.key];
            const w = (v / total) * 100;
            if (w <= 0) return null;
            return (
              <div
                key={s.key}
                className="h-full transition-all"
                style={{ width: `${w}%`, background: s.color }}
                title={`${s.label}: ${fmtMs(v)}`}
              />
            );
          })}
        </div>
        {!compact && (
          <div className="flex flex-wrap gap-x-4 gap-y-1 pt-0.5">
            {SEGMENTS.map((s) => (
              <span key={s.key} className="flex items-center gap-1.5 text-[11px] text-muted">
                <span className="h-2 w-2 rounded-sm" style={{ background: s.color }} />
                {s.label}
                <span className="tabular text-faint">{fmtMs(latencyMs[s.key])}</span>
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Stat({
  icon: Icon,
  label,
  value,
  unit,
  tint,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  unit?: string;
  tint: string;
}) {
  return (
    <div className="rounded-xl border border-line bg-ink-2/50 p-2.5">
      <div className="mb-1 flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-faint">
        <Icon className="h-3 w-3" />
        {label}
      </div>
      <div className="tabular text-[15px] font-semibold" style={{ color: tint }}>
        {value}
        {unit ? <span className="ml-0.5 text-[11px] text-faint">{unit}</span> : null}
      </div>
    </div>
  );
}

export function ColdWarmBadge({ cold }: { cold: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium",
        cold
          ? "bg-[color-mix(in_oklab,var(--color-fp16)_16%,transparent)] text-fp16"
          : "bg-[color-mix(in_oklab,var(--color-warn)_16%,transparent)] text-warn",
      )}
      title={cold ? "Weights were loaded into VRAM for this request" : "Weights already resident — warm cache hit"}
    >
      {cold ? <Snowflake className="h-3 w-3" /> : <Flame className="h-3 w-3" />}
      {cold ? "cold start" : "warm"}
    </span>
  );
}
