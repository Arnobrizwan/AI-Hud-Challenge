"use client";

import { useQuery, useMutation } from "convex/react";
import { api } from "@/convex/_generated/api";
import type { Id } from "@/convex/_generated/dataModel";
import { timeAgo, faviconOf } from "@/lib/utils";
import { Bookmark, Trash2, ExternalLink } from "lucide-react";

export function BookmarksView() {
  const bookmarks = useQuery(api.bookmarks.list);
  const remove = useMutation(api.bookmarks.remove);
  const now = Date.now();

  return (
    <div className="h-full overflow-y-auto hud-scroll px-4 sm:px-8 py-6">
      <div className="max-w-3xl mx-auto">
        <header className="mb-5 flex items-center gap-2">
          <Bookmark className="w-5 h-5 text-cyan" />
          <h1 className="hud-title text-2xl text-ink">Saved signal</h1>
          {bookmarks && (
            <span className="hud-chip ml-2">{bookmarks.length}</span>
          )}
        </header>

        {bookmarks === undefined ? (
          <div className="hud-label py-20 text-center">loading…</div>
        ) : bookmarks.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-24 text-center gap-2">
            <Bookmark className="w-8 h-8 text-ink-faint" />
            <p className="text-ink-dim text-sm max-w-xs">
              Nothing saved yet. Hit the bookmark icon on any card — saved items
              resurface in your stream on the schedule you set in Config.
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {bookmarks.map((b) => (
              <article key={b.bookmarkId} className="hud-panel hud-clip p-4 flex gap-3 group">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1.5">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={faviconOf(b.url)} alt="" width={14} height={14} className="rounded-sm opacity-80" />
                    <span className="text-[11px] text-ink-dim">{b.sourceName}</span>
                    <span className="text-[10px] text-ink-faint ml-auto">saved {timeAgo(b.savedAt, now)} ago</span>
                  </div>
                  <a
                    href={b.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hud-title text-[15px] text-ink hover:text-cyan-soft transition-colors"
                  >
                    {b.title}
                  </a>
                  <p className="text-[12.5px] text-ink-dim leading-relaxed mt-1 line-clamp-2">{b.summary}</p>
                  <div className="flex items-center gap-1.5 mt-2">
                    {b.topics.slice(0, 3).map((t) => (
                      <span key={t} className="text-[9px] text-ink-faint uppercase tracking-wider px-1.5 py-0.5 rounded border border-[var(--line)]">
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="flex flex-col gap-1 shrink-0">
                  <a href={b.url} target="_blank" rel="noopener noreferrer" className="p-1.5 rounded text-ink-faint hover:text-cyan transition-colors" title="Open">
                    <ExternalLink className="w-4 h-4" />
                  </a>
                  <button
                    onClick={() => remove({ itemId: b._id as Id<"items"> })}
                    className="p-1.5 rounded text-ink-faint hover:text-rose transition-colors"
                    title="Remove"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
