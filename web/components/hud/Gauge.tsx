"use client";

import { cn } from "@/lib/utils";

/** Circular HUD gauge showing a 0..1 value as a percentage ring. */
export function Gauge({
  value,
  label,
  size = 46,
  color = "var(--cyan)",
}: {
  value: number;
  label?: string;
  size?: number;
  color?: string;
}) {
  const v = Math.max(0, Math.min(1, value));
  const stroke = 3.5;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.round(v * 100);

  return (
    <div className="flex flex-col items-center justify-center" style={{ width: size }}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke="var(--line)"
            strokeWidth={stroke}
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth={stroke}
            strokeDasharray={c}
            strokeDashoffset={c * (1 - v)}
            strokeLinecap="round"
            style={{ transition: "stroke-dashoffset 0.6s ease", filter: `drop-shadow(0 0 4px ${color})` }}
          />
        </svg>
        <span
          className="hud-title font-semibold absolute inset-0 flex items-center justify-center"
          style={{ fontSize: size * 0.26, color }}
        >
          {pct}
        </span>
      </div>
      {label && <span className="hud-label !text-[8px] mt-0.5">{label}</span>}
    </div>
  );
}

/** Horizontal mini feature bar for the score breakdown popover. */
export function FeatureBar({
  label,
  value,
  color = "var(--cyan-soft)",
}: {
  label: string;
  value: number;
  color?: string;
}) {
  const v = Math.max(0, Math.min(1, value));
  return (
    <div className="flex items-center gap-2">
      <span className="hud-label !text-[8px] w-16 shrink-0 !text-ink-faint">{label}</span>
      <div className="flex-1 h-1 rounded-full bg-[var(--line)] overflow-hidden">
        <div
          className={cn("h-full rounded-full")}
          style={{ width: `${v * 100}%`, background: color }}
        />
      </div>
      <span className="text-[9px] text-ink-dim w-6 text-right tabular-nums">
        {Math.round(v * 100)}
      </span>
    </div>
  );
}
