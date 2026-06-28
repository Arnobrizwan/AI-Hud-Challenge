"use client";

import { useState } from "react";
import { useMutation, useAction } from "convex/react";
import { api } from "@/convex/_generated/api";
import type { Id } from "@/convex/_generated/dataModel";
import { cn, timeAgo, faviconOf } from "@/lib/utils";
import { Gauge, FeatureBar } from "./Gauge";
import {
  Bookmark, BookmarkCheck, ThumbsUp, ThumbsDown, VolumeX, EyeOff,
  ExternalLink, Sparkles, Info, MessageSquare, Flame,
} from "lucide-react";

export interface FeedCard {
  _id: string;
  title: string;
  url: string;
  sourceId: string;
  sourceName: string;
  kind: string;
  summary: string;
  hasAbstractive: boolean;
  image: string | null;
  publishedAt: number;
  topics: string[];
  entities: string[];
  contentType: string;
  engagement: { points: number; comments: number };
  score: number;
  lane: "focus" | "trending";
  breakdown: {
    recency: number; sourceWeight: number; topicalMatch: number;
    novelty: number; velocity: number; popularity: number;
  };
  bookmarked: boolean;
  related: number;
  resurfaced?: boolean;
  trendlet?: "new" | "updated" | null;
  flagged?: boolean;
  entityLinks?: { name: string; qid: string }[];
}

export function NewsCard({
  card,
  now,
  onHide,
}: {
  card: FeedCard;
  now: number;
  onHide?: (id: string) => void;
}) {
  const record = useMutation(api.feedback.record);
  const toggleBookmark = useMutation(api.bookmarks.toggle);
  const enhance = useAction(api.items.enhanceSummary);

  const [bookmarked, setBookmarked] = useState(card.bookmarked);
  const [summary, setSummary] = useState(card.summary);
  const [isAbstractive, setIsAbstractive] = useState(card.hasAbstractive);
  const [enhancing, setEnhancing] = useState(false);
  const [showBreakdown, setShowBreakdown] = useState(false);
  const [voted, setVoted] = useState<null | "up" | "down">(null);

  const id = card._id as Id<"items">;

  async function onEnhance() {
    if (enhancing || isAbstractive) return;
    setEnhancing(true);
    try {
      const res = await enhance({ itemId: id });
      if (res.summary) {
        setSummary(res.summary);
        setIsAbstractive(true);
      }
    } finally {
      setEnhancing(false);
    }
  }

  const laneFocus = card.lane === "focus";

  return (
    <article className="hud-panel hud-clip p-4 group relative overflow-hidden">
      {/* lane accent edge */}
      <span
        className="absolute left-0 top-0 bottom-0 w-[3px]"
        style={{ background: laneFocus ? "var(--cyan)" : "var(--violet)" }}
      />

      <div className="flex gap-3">
        {/* left: text */}
        <div className="flex-1 min-w-0">
          {/* meta row */}
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={faviconOf(card.url)}
              alt=""
              width={14}
              height={14}
              className="rounded-sm opacity-80"
            />
            <span className="text-[11px] text-ink-dim font-medium truncate max-w-[140px]">
              {card.sourceName}
            </span>
            <span className={cn("hud-chip", laneFocus ? "" : "hud-chip-violet")}>
              {laneFocus ? "Focus" : "Trending"}
            </span>
            {card.resurfaced && <span className="hud-chip hud-chip-amber">From bookmarks</span>}
            {card.trendlet === "new" && <span className="hud-chip">▲ New</span>}
            {card.trendlet === "updated" && <span className="hud-chip hud-chip-amber">↻ Updated</span>}
            {card.engagement.points > 50 && (
              <span className="hud-chip hud-chip-rose">
                <Flame className="w-2.5 h-2.5" /> {card.engagement.points}
              </span>
            )}
            <span className="text-[10px] text-ink-faint ml-auto tabular-nums">
              {timeAgo(card.publishedAt, now)}
            </span>
          </div>

          {/* title */}
          <a
            href={card.url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={() => record({ itemId: id, action: "click" })}
            className="block hud-title text-[15px] leading-snug text-ink hover:text-cyan-soft transition-colors"
          >
            {card.title}
          </a>

          {/* summary (hidden when it merely repeats the title, e.g. link-only posts) */}
          {summary &&
            summary.trim().slice(0, 60).toLowerCase() !==
              card.title.trim().slice(0, 60).toLowerCase() && (
              <p className="text-[12.5px] text-ink-dim leading-relaxed mt-1.5 line-clamp-2">
                {summary}
              </p>
            )}

          {/* footer chips + actions */}
          <div className="flex items-center gap-1.5 mt-2.5 flex-wrap">
            {card.topics.slice(0, 3).map((t) => (
              <span key={t} className="text-[9px] text-ink-faint uppercase tracking-wider px-1.5 py-0.5 rounded border border-[var(--line)]">
                {t}
              </span>
            ))}
            {card.related > 0 && (
              <span className="text-[9px] text-cyan-soft px-1.5 py-0.5">+{card.related} related</span>
            )}
            {card.engagement.comments > 0 && (
              <span className="text-[9px] text-ink-faint flex items-center gap-0.5">
                <MessageSquare className="w-2.5 h-2.5" /> {card.engagement.comments}
              </span>
            )}

            <div className="flex items-center gap-0.5 ml-auto opacity-60 group-hover:opacity-100 transition-opacity">
              {isAbstractive ? (
                <span className="hud-chip hud-chip-violet"><Sparkles className="w-2.5 h-2.5" /> AI</span>
              ) : (
                <IconBtn title="AI summary (uses your key)" onClick={onEnhance} active={enhancing}>
                  <Sparkles className={cn("w-3.5 h-3.5", enhancing && "animate-pulse")} />
                </IconBtn>
              )}
              <IconBtn
                title="Why this?"
                onClick={() => setShowBreakdown((s) => !s)}
                active={showBreakdown}
              >
                <Info className="w-3.5 h-3.5" />
              </IconBtn>
              <IconBtn
                title={voted === "up" ? "Liked" : "More like this"}
                onClick={() => {
                  setVoted("up");
                  record({ itemId: id, action: "up" });
                }}
                active={voted === "up"}
              >
                <ThumbsUp className="w-3.5 h-3.5" />
              </IconBtn>
              <IconBtn
                title="Less like this"
                onClick={() => {
                  setVoted("down");
                  record({ itemId: id, action: "down" });
                }}
                active={voted === "down"}
              >
                <ThumbsDown className="w-3.5 h-3.5" />
              </IconBtn>
              <IconBtn
                title="Mute source"
                onClick={() => {
                  record({ itemId: id, action: "mute_source" });
                  onHide?.(card._id);
                }}
              >
                <VolumeX className="w-3.5 h-3.5" />
              </IconBtn>
              <IconBtn
                title="Not interested"
                onClick={() => {
                  record({ itemId: id, action: "not_interested" });
                  onHide?.(card._id);
                }}
              >
                <EyeOff className="w-3.5 h-3.5" />
              </IconBtn>
              <IconBtn
                title={bookmarked ? "Remove bookmark" : "Save"}
                onClick={async () => {
                  setBookmarked((b) => !b);
                  await toggleBookmark({ itemId: id });
                }}
                active={bookmarked}
              >
                {bookmarked ? <BookmarkCheck className="w-3.5 h-3.5" /> : <Bookmark className="w-3.5 h-3.5" />}
              </IconBtn>
              <a
                href={card.url}
                target="_blank"
                rel="noopener noreferrer"
                className="p-1.5 rounded text-ink-faint hover:text-cyan transition-colors"
                title="Open"
              >
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>
          </div>

          {/* score breakdown */}
          {showBreakdown && (
            <div className="mt-3 pt-3 border-t border-[var(--line)] grid grid-cols-2 gap-x-4 gap-y-1.5">
              <FeatureBar label="Recency" value={card.breakdown.recency} />
              <FeatureBar label="Topical" value={card.breakdown.topicalMatch} color="var(--cyan)" />
              <FeatureBar label="Popularity" value={card.breakdown.popularity} color="var(--violet)" />
              <FeatureBar label="Velocity" value={card.breakdown.velocity} color="var(--violet)" />
              <FeatureBar label="Novelty" value={card.breakdown.novelty} />
              <FeatureBar label="Source" value={card.breakdown.sourceWeight} />
            </div>
          )}
        </div>

        {/* right: gauges */}
        <div className="hidden md:flex flex-col items-center gap-2 shrink-0 pt-0.5">
          <Gauge value={card.breakdown.topicalMatch} label="Focus" size={44} />
          <Gauge value={card.breakdown.popularity} label="Pop" size={44} color="var(--violet)" />
        </div>
      </div>
    </article>
  );
}

function IconBtn({
  children,
  title,
  onClick,
  active,
}: {
  children: React.ReactNode;
  title: string;
  onClick: () => void;
  active?: boolean;
}) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={cn(
        "p-1.5 rounded transition-colors",
        active ? "text-cyan bg-[rgba(46,230,230,0.12)]" : "text-ink-faint hover:text-cyan",
      )}
    >
      {children}
    </button>
  );
}
