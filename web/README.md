# ◈ HUD — High-Signal Personal News Feed

A heads-up display for the internet. HUD pulls from **HackerNews, AI newsletters
(TLDR AI, AI News, The Rundown), subreddits, and X**, dedupes the noise, ranks
for **your focus areas blended with what's actually breaking**, and streams it as
an **auto-scrolling HUD** — with bookmarks that resurface, bring-your-own AI keys,
and a live operator dashboard for the ranking pipeline.

Built for the **Open HUD Challenge**. Stack: **Next.js 16 (App Router) · Convex ·
TypeScript · Tailwind v4**, deployed on **Vercel**.

→ See **[PIPELINE.md](./PIPELINE.md)** for how every stage of the AI/ML pipeline is implemented.

---

## What it does

- **Auto-scrolling stream** with adjustable speed, pause/hover, and `J/K/space` keyboard controls.
- **Focus × Trending mix** on a single slider; every card is tagged `Focus` or `Trending` and shows *why* it surfaced (recency / topical / popularity / velocity / novelty / source breakdown).
- **Transparent ranking**: a tunable weighted score; popularity = real HN points / Reddit ups + cross-source velocity.
- **Dedup → events**: MinHash + LSH clusters syndicated copies into one card with "+N related".
- **Bookmarks** that **resurface** on your schedule; **feedback** (👍/👎/mute/not-interested) that re-ranks live.
- **Breaking ticker**: high-velocity / high-engagement events, interest-biased.
- **Bring your own AI key** (OpenAI / Anthropic) for grounded abstractive summaries — encrypted at rest; app stays fully functional without one.
- **Pipeline dashboard**: throughput, source health, **hot-reload ranking weights**, eval metrics (P@K, nDCG, coverage, novelty, diversity, dup-F1), and per-run stage timings.
- **Multi-user** via Convex Auth (password + one-tap guest), with per-user prefs/bookmarks/keys fully isolated.

## Architecture

```
Next.js (Vercel)                Convex (reactive backend)
┌────────────────────┐          ┌─────────────────────────────────┐
│ app/feed           │  live    │ crons → pipeline.runPipeline     │
│ app/bookmarks      │ queries  │   ingest→normalize→enrich→dedup  │
│ app/settings       │◀────────▶│   →rank→summarize→persist        │
│ app/dashboard      │ mutations│ feed.getFeed (read-time ranking) │
│ components/hud/*    │          │ auth · feedback · bookmarks · eval│
└────────────────────┘          └─────────────────────────────────┘
        lib/pipeline/*  ← pure, reusable, stage-isolated TS (shared)
```

- **Pipeline** (`lib/pipeline/`) is pure TS, decoupled from the app — see PIPELINE.md.
- **Backend** (`convex/`) orchestrates + persists; ranking runs read-time so prefs/weights apply instantly.
- The legacy Python microservices in the parent repo are kept as the AI-engineer **reference**; this app is self-contained and Vercel-native.

## Local development

```bash
cd web
npm install
npx convex dev          # provisions/links a Convex dev deployment, runs codegen
# in another shell:
npm run dev             # Next.js on http://localhost:3000
```

First-time backend setup (once per deployment):

```bash
npx @convex-dev/auth --web-server-url http://localhost:3000   # JWT keys + SITE_URL
npx convex env set KEY_ENCRYPTION_SECRET "$(openssl rand -base64 32)"  # BYO-key encryption
npx convex run sources:seed          # seed the source catalog + default ranking config
npx convex run pipeline:runPipeline '{"trigger":"manual"}'   # pull the first batch
```

The pipeline then runs automatically every 20 minutes (`convex/crons.ts`).

## Environment variables

| Where | Var | Purpose |
|---|---|---|
| `.env.local` (Next.js) | `NEXT_PUBLIC_CONVEX_URL` | Convex client URL (written by `convex dev`) |
| Convex deployment | `KEY_ENCRYPTION_SECRET` | AES-GCM secret for BYO API keys |
| Convex deployment | `SITE_URL`, `JWT_PRIVATE_KEY`, `JWKS` | Convex Auth (set by `@convex-dev/auth`) |

## Deploy (Vercel + Convex)

```bash
npx convex deploy        # push functions to a Convex prod deployment
# Set the same env vars on prod, then seed + run:
#   npx convex env set KEY_ENCRYPTION_SECRET ... --prod
#   npx @convex-dev/auth --prod --web-server-url https://<your-vercel-domain>
vercel --prod            # NEXT_PUBLIC_CONVEX_URL = prod Convex URL
```

## Project layout

```
web/
├─ app/                       routes: /, /signin, /feed, /bookmarks, /settings, /dashboard
├─ components/hud/            AppFrame, FeedView, NewsCard, Gauge, BreakingTicker, Settings/Bookmarks/Dashboard views
├─ lib/pipeline/              ingest · normalize · enrich · dedup · rank · summarize (pure TS)
├─ convex/                    schema, auth, pipeline orchestrator, crons, feed/feedback/bookmarks/eval/dashboard
└─ proxy.ts                   Next 16 auth middleware (route gating)
```

Built with Next.js, Convex, and Claude Code.
