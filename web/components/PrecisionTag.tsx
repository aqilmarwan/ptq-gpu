import type { Precision } from "@/lib/types";
import { precisionColor } from "@/lib/variants";
import { cn } from "@/lib/utils";

/** Color-coded precision chip — the visual key to the whole studio. */
export function PrecisionTag({
  precision,
  size = "sm",
  className,
}: {
  precision: Precision;
  size?: "sm" | "xs";
  className?: string;
}) {
  const color = precisionColor(precision);
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md font-mono font-medium tabular",
        size === "sm" ? "px-2 py-0.5 text-[11px]" : "px-1.5 py-0.5 text-[10px]",
        className,
      )}
      style={{
        color,
        background: `color-mix(in oklab, ${color} 14%, transparent)`,
        boxShadow: `inset 0 0 0 1px color-mix(in oklab, ${color} 35%, transparent)`,
      }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
      {precision}
    </span>
  );
}
