"use client";

import { useState } from "react";
import { useQuery } from "convex/react";
import { api } from "@/convex/_generated/api";
import { Zap, X } from "lucide-react";

/** A glanceable "breaking" strip — high-velocity / high-engagement events. */
export function BreakingTicker() {
  const breaking = useQuery(api.notifications.getBreaking);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());

  const items = (breaking ?? []).filter((b) => !dismissed.has(b._id));
  if (items.length === 0) return null;

  return (
    <div className="px-4 sm:px-6 py-2 border-b border-[var(--line)] bg-[rgba(255,93,115,0.05)] shrink-0 flex items-center gap-3 overflow-hidden">
      <span className="hud-chip hud-chip-rose shrink-0">
        <Zap className="w-2.5 h-2.5" /> Breaking
      </span>
      <div className="flex items-center gap-6 overflow-x-auto hud-noscroll flex-1">
        {items.map((b) => (
          <a
            key={b._id}
            href={b.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 whitespace-nowrap text-[12px] text-ink-dim hover:text-rose transition-colors shrink-0"
          >
            {b.onFocus && <span className="hud-chip !py-0.5">focus</span>}
            <span className="truncate max-w-[340px]">{b.title}</span>
            <span className="text-ink-faint text-[10px]">· {b.sourceName} · {b.points}▲</span>
          </a>
        ))}
      </div>
      <button onClick={() => setDismissed(new Set(items.map((b) => b._id)))} className="text-ink-faint hover:text-ink-dim shrink-0">
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}
