import type { Variant } from "./types";
import { mulberry32 } from "./utils";

/**
 * Procedural placeholder "renderer".
 *
 * Generates a deterministic abstract artwork as an inline SVG data URL.
 * Composition is a pure function of (prompt, seed) — so Compare mode, which
 * fixes both, yields the SAME composition across variants. Only the quality
 * overlay (film grain / posterisation) changes with precision, which makes the
 * speed↔quality tradeoff *visible* even before the real GPU backend is wired in.
 *
 * When FastAPI is serving, `result.imageUrl` is a real PNG and this is unused.
 */

const PALETTES: string[][] = [
  ["#1e1b4b", "#7c3aed", "#22d3ee", "#f0abfc"],
  ["#0c4a6e", "#0ea5e9", "#5eead4", "#fde68a"],
  ["#4a044e", "#db2777", "#fb7185", "#fdba74"],
  ["#052e16", "#16a34a", "#a3e635", "#fde047"],
  ["#1c1917", "#f97316", "#fbbf24", "#fca5a5"],
  ["#0f172a", "#6366f1", "#38bdf8", "#e0e7ff"],
];

function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function renderArtwork(prompt: string, seed: number, variant: Variant): string {
  const rng = mulberry32((hashString(prompt) ^ (seed * 2654435761)) >>> 0);
  const palette = PALETTES[Math.floor(rng() * PALETTES.length)];

  const blobs = Array.from({ length: 5 }, (_, i) => {
    const cx = Math.round(rng() * 100);
    const cy = Math.round(rng() * 100);
    const r = 30 + Math.round(rng() * 45);
    const color = palette[(i + 1) % palette.length];
    const op = (0.5 + rng() * 0.45).toFixed(2);
    return `<radialGradient id="g${i}" cx="${cx}%" cy="${cy}%" r="${r}%">
        <stop offset="0%" stop-color="${color}" stop-opacity="${op}"/>
        <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
      </radialGradient>
      <rect width="768" height="768" fill="url(#g${i})"/>`;
  }).join("");

  // grain rises as quality falls — FP8 looks visibly grainier than FP16
  const grain = ((100 - variant.quality) / 100) * 0.22;
  // a soft diagonal light streak for a "rendered" feel
  const streakAngle = Math.round(rng() * 360);

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="768" height="768" viewBox="0 0 768 768">
    <defs>
      <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="${palette[0]}"/>
        <stop offset="100%" stop-color="#05050a"/>
      </linearGradient>
      <linearGradient id="streak" gradientTransform="rotate(${streakAngle} 0.5 0.5)">
        <stop offset="0%" stop-color="#ffffff" stop-opacity="0"/>
        <stop offset="50%" stop-color="#ffffff" stop-opacity="0.10"/>
        <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
      </linearGradient>
      <filter id="soft"><feGaussianBlur stdDeviation="18"/></filter>
      <filter id="grain">
        <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" stitchTiles="stitch"/>
        <feColorMatrix type="saturate" values="0"/>
        <feComponentTransfer><feFuncA type="linear" slope="${grain.toFixed(3)}"/></feComponentTransfer>
      </filter>
    </defs>
    <rect width="768" height="768" fill="url(#bg)"/>
    <g filter="url(#soft)">${blobs}</g>
    <rect width="768" height="768" fill="url(#streak)"/>
    <rect width="768" height="768" filter="url(#grain)"/>
  </svg>`;

  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}
