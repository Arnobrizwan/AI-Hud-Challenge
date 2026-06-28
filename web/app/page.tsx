import Link from "next/link";
import { Radar, Bookmark, SlidersHorizontal, KeyRound, Gauge, Layers } from "lucide-react";

const FEATURES = [
  { icon: Radar, title: "Auto-scrolling stream", body: "A glanceable HUD that flows on its own — speed you control." },
  { icon: Layers, title: "Focus × Trending mix", body: "Blend your topics with what's breaking, on a single slider." },
  { icon: Gauge, title: "Transparent ranking", body: "Every card shows why it surfaced: recency, velocity, popularity." },
  { icon: Bookmark, title: "Bookmarks that resurface", body: "Save signal; it re-enters the stream on your schedule." },
  { icon: KeyRound, title: "Bring your own AI key", body: "Plug in OpenAI / Anthropic for abstractive summaries. Encrypted." },
  { icon: SlidersHorizontal, title: "Operator dashboard", body: "Tune ranking weights live and watch the eval metrics move." },
];

export default function Home() {
  return (
    <main className="flex flex-1 flex-col">
      {/* top bar */}
      <header className="flex items-center justify-between px-6 sm:px-10 py-5">
        <span className="hud-title text-cyan-soft tracking-[0.3em] text-sm">◈ HUD</span>
        <Link href="/signin" className="hud-btn">
          Launch HUD
        </Link>
      </header>

      {/* hero */}
      <section className="flex flex-col items-center text-center px-6 pt-16 pb-20 sm:pt-24">
        <span className="hud-chip mb-6">Open HUD Challenge · High-Signal News</span>
        <h1 className="hud-title text-4xl sm:text-6xl font-bold text-ink max-w-3xl leading-[1.05]">
          A heads-up display
          <br />
          <span className="text-cyan hud-glow-text">for the internet.</span>
        </h1>
        <p className="text-ink-dim max-w-xl mt-6 text-base sm:text-lg leading-relaxed">
          Pulls HackerNews, AI newsletters, subreddits and X — dedupes the noise,
          ranks for <span className="text-ink">your</span> focus and what&apos;s
          actually breaking, and streams it as an auto-scrolling HUD.
        </p>
        <div className="flex gap-3 mt-9">
          <Link href="/signin" className="hud-btn text-sm px-6 py-3">
            Enter the HUD →
          </Link>
          <Link
            href="/signin"
            className="hud-btn text-sm px-6 py-3 !border-[rgba(160,107,255,0.5)] !text-violet-soft !bg-[rgba(160,107,255,0.1)]"
          >
            Try as guest
          </Link>
        </div>
      </section>

      {/* feature grid */}
      <section className="px-6 sm:px-10 pb-24 max-w-5xl mx-auto w-full">
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {FEATURES.map((f) => (
            <div key={f.title} className="hud-panel hud-clip p-5">
              <f.icon className="w-5 h-5 text-cyan mb-3" />
              <h3 className="text-ink font-semibold text-sm mb-1">{f.title}</h3>
              <p className="text-ink-dim text-xs leading-relaxed">{f.body}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="px-6 sm:px-10 py-6 text-center text-[11px] text-ink-faint border-t border-[var(--line)]">
        Built for the Open HUD Challenge · Next.js + Convex · Bring-your-own-AI-key
      </footer>
    </main>
  );
}
