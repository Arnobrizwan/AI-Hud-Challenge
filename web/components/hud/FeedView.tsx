"use client";

import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { useQuery, useMutation } from "convex/react";
import { api } from "@/convex/_generated/api";
import { NewsCard, type FeedCard } from "./NewsCard";
import { BreakingTicker } from "./BreakingTicker";
import { cn } from "@/lib/utils";
import { Pause, Play, Gauge as GaugeIcon, Layers, X } from "lucide-react";

const TOPICS = ["all", "ai", "llm", "agents", "startups", "programming", "security", "open-source", "science", "hardware"];

export function FeedView() {
  const prefs = useQuery(api.prefs.getPrefs);
  const updatePrefs = useMutation(api.prefs.updatePrefs);
  const markResurfaced = useMutation(api.bookmarks.markResurfaced);

  const [topic, setTopic] = useState("all");
  const feed = useQuery(api.feed.getFeed, { topic, limit: 60 });
  const resurfacing = useQuery(api.bookmarks.resurfacing);

  // local mirror of prefs for snappy sliders; debounced write-back
  const [speed, setSpeed] = useState(26);
  const [mix, setMix] = useState(0.6);
  const [paused, setPaused] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [now, setNow] = useState(() => Date.now());
  const [resurfaceDismissed, setResurfaceDismissed] = useState(false);

  useEffect(() => {
    if (prefs) {
      setSpeed(prefs.autoScrollSpeed);
      setMix(prefs.focusVsPopularMix);
    }
  }, [prefs]);

  // tick "now" for live ages
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(t);
  }, []);

  const scrollRef = useRef<HTMLDivElement>(null);

  // debounced pref persistence
  const writeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const persist = useCallback(
    (patch: { autoScrollSpeed?: number; focusVsPopularMix?: number }) => {
      if (writeTimer.current) clearTimeout(writeTimer.current);
      writeTimer.current = setTimeout(() => updatePrefs(patch), 450);
    },
    [updatePrefs],
  );

  const cards: FeedCard[] = useMemo(
    () => (feed?.items ?? []).filter((c) => !hidden.has(c._id)) as FeedCard[],
    [feed, hidden],
  );

  // seamless auto-scroll loop
  const effectivePaused = paused || hovered || speed <= 0;
  useEffect(() => {
    let raf = 0;
    let last: number | null = null;
    const step = (t: number) => {
      if (last == null) last = t;
      const dt = (t - last) / 1000;
      last = t;
      const el = scrollRef.current;
      if (el && !effectivePaused) {
        el.scrollTop += speed * dt;
        const half = el.scrollHeight / 2;
        if (half > 0 && el.scrollTop >= half) el.scrollTop -= half;
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [effectivePaused, speed]);

  // keyboard controls
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      const el = scrollRef.current;
      if (e.key === "j" && el) el.scrollTop += 140;
      else if (e.key === "k" && el) el.scrollTop -= 140;
      else if (e.key === "." || e.key === " ") {
        e.preventDefault();
        setPaused((p) => !p);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const onHide = useCallback((id: string) => {
    setHidden((h) => new Set(h).add(id));
  }, []);

  // duplicate the list for a seamless wrap when there's enough content
  const loopCards = cards.length > 6 ? [...cards, ...cards] : cards;

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* control deck */}
      <div className="px-4 sm:px-6 py-3 border-b border-[var(--line)] flex items-center gap-3 flex-wrap shrink-0">
        {/* topic tabs */}
        <div className="flex items-center gap-1 overflow-x-auto hud-noscroll max-w-full">
          {TOPICS.map((t) => (
            <button
              key={t}
              onClick={() => setTopic(t)}
              className={cn(
                "px-2.5 py-1 rounded-md text-[10px] font-semibold uppercase tracking-wider whitespace-nowrap transition-all border",
                topic === t
                  ? "text-cyan border-[var(--line-bright)] bg-[rgba(46,230,230,0.1)]"
                  : "text-ink-faint border-transparent hover:text-ink-dim",
              )}
            >
              {t === "all" ? "◇ All" : t}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-4 ml-auto">
          {/* mix slider */}
          <div className="flex items-center gap-2">
            <Layers className="w-3.5 h-3.5 text-violet" />
            <input
              type="range" min={0} max={1} step={0.05} value={mix}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                setMix(v);
                persist({ focusVsPopularMix: v });
              }}
              className="hud-range w-24"
              title="Focus ↔ Trending mix"
            />
            <span className="hud-label !text-[8px] w-20">
              {mix >= 0.6 ? "focus-led" : mix <= 0.4 ? "trend-led" : "balanced"}
            </span>
          </div>

          {/* speed slider */}
          <div className="flex items-center gap-2">
            <GaugeIcon className="w-3.5 h-3.5 text-cyan" />
            <input
              type="range" min={0} max={80} step={2} value={speed}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10);
                setSpeed(v);
                persist({ autoScrollSpeed: v });
              }}
              className="hud-range w-20"
              title="Auto-scroll speed"
            />
          </div>

          <button
            onClick={() => setPaused((p) => !p)}
            className="hud-btn !py-1.5 !px-3"
            title="Pause / play (space)"
          >
            {effectivePaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      <BreakingTicker />

      {/* resurfaced banner */}
      {resurfacing && resurfacing.length > 0 && !resurfaceDismissed && (
        <div className="px-4 sm:px-6 py-2 border-b border-[var(--line)] flex items-center gap-2 bg-[rgba(255,181,71,0.05)] shrink-0">
          <span className="hud-chip hud-chip-amber">From bookmarks</span>
          <a
            href={resurfacing[0].url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={() => markResurfaced({ bookmarkId: resurfacing[0].bookmarkId })}
            className="text-[12px] text-ink-dim hover:text-amber truncate flex-1"
          >
            {resurfacing[0].title}
          </a>
          <button onClick={() => setResurfaceDismissed(true)} className="text-ink-faint hover:text-ink-dim">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* stream */}
      <div
        ref={scrollRef}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        className="flex-1 min-h-0 overflow-y-auto hud-noscroll px-4 sm:px-6 py-4"
      >
        {feed === undefined ? (
          <div className="flex items-center justify-center h-40 hud-label">acquiring signal…</div>
        ) : cards.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="flex flex-col gap-3 max-w-3xl mx-auto">
            {loopCards.map((c, i) => (
              <NewsCard key={`${c._id}-${i}`} card={c} now={now} onHide={onHide} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center gap-3 py-20">
      <span className="hud-title text-cyan text-3xl hud-pulse">◈</span>
      <p className="text-ink-dim text-sm max-w-xs">
        No signals in this lane yet. The pipeline pulls fresh content every 20
        minutes — try the <span className="text-cyan">All</span> tab, or run the
        pipeline from the Pipeline dashboard.
      </p>
    </div>
  );
}
