# ◈ HUD — High-Signal Personal News Feed

A heads-up display for the internet. HUD pulls from **HackerNews, AI newsletters
(TLDR AI, AI News, The Rundown), subreddits, and X**, dedupes the noise, ranks
for **your focus areas blended with what's actually breaking**, and streams it as
an **auto-scrolling HUD** — with bookmarks that resurface, bring-your-own AI keys,
and a live operator dashboard for the ranking pipeline.

Built for the **Open HUD Challenge**. Stack: **Next.js 16 (App Router) · Convex ·
TypeScript · Tailwind v4**, deployed on **Vercel**. The app lives in [`web/`](./web).

**▶ Live:** **https://hud-news.vercel.app** — sign in as **guest** to try instantly.

**Docs:** [web/PIPELINE.md](./web/PIPELINE.md) (every AI/ML stage) · [web/PRODUCT.md](./web/PRODUCT.md) (six-pager) · [web/FEATURE_DICTIONARY.md](./web/FEATURE_DICTIONARY.md) · [web/RUNBOOK.md](./web/RUNBOOK.md) (runbook + ADRs + privacy) · [web/BUILD_NOTES.md](./web/BUILD_NOTES.md) · [web/design/hud-design.html](./web/design/hud-design.html) (browser mockup)

---

## What it does

- **Auto-scrolling stream** with adjustable speed, pause/hover, and `J/K/space` keyboard controls.
- **Focus × Trending mix** on a single slider; every card is tagged `Focus` or `Trending` and shows *why* it surfaced (recency / topical / popularity / velocity / novelty / source breakdown).
- **Transparent ranking**: a tunable weighted score; popularity = real HN points / Reddit ups + cross-source velocity.
- **Dedup → events**: MinHash + LSH clusters syndicated copies into one card with "+N related", with a SimHash + cosine second opinion on borderline pairs.
- **Bookmarks** that **resurface** on your schedule; **feedback** (👍/👎/mute/not-interested) that re-ranks live.
- **Breaking ticker**: high-velocity / high-engagement events, interest-biased, with timezone-aware quiet hours.
- **Bring your own AI key** (OpenAI / Anthropic) for grounded abstractive summaries — encrypted at rest; app stays fully functional without one.
- **Pipeline dashboard**: throughput, source health, **hot-reload ranking weights**, eval metrics (P@K, nDCG, coverage, novelty, diversity, dup-F1), and per-run stage timings.
- **Multi-user** via Convex Auth (password + one-tap guest), with per-user prefs/bookmarks/keys fully isolated.

## Sources & status

Configured in `web/convex/seedData.ts`; manage live from the dashboard.

| Source | Status | Notes |
|---|---|---|
| **HackerNews** | ✅ live | Algolia API — real points + comments |
| **TLDR AI** | ✅ live | RSS |
| **AI News (smol.ai)** | ✅ live | RSS `news.smol.ai/rss.xml` |
| **The Rundown AI** | ✅ live | beehiiv RSS |
| **Latent Space** | ✅ live | RSS (bonus AI newsletter) |
| **Daring Fireball** | ✅ live | **JSON Feed** (jsonfeed.org spec) |
| OpenAI / DeepMind / Hugging Face blogs, TechCrunch, Ars Technica, The Verge, VentureBeat | ✅ live | RSS |
| **Subreddits** | ⏸️ disabled by choice | Reddit `429`-blocks datacenter IPs (Convex/Vercel egress), so only ~1/5 worked. Left in the catalog (off); re-enable + add Reddit OAuth (`REDDIT_CLIENT_ID`/`SECRET`) for reliable coverage. |
| **X / Twitter** | ⏸️ off | No free X API; nitter mirrors are mostly dead. Re-enable by pointing the source at a working RSS-bridge URL or wiring an X API v2 bearer token. |

The live mix is **HackerNews + AI newsletters + tech RSS + JSON Feed**. The feed shows items from the last **72h**, so daily newsletters appear alongside high-frequency sources like HackerNews. Subreddit and X connectors are implemented and remain in the catalog (disabled) — flip them on from the dashboard once credentials/a working bridge are available.

Adapters live in `web/lib/pipeline/ingest/` (`rss`, `hackernews`, `jsonfeed`). Before each
request, **robots.txt** is fetched per-origin (cached) and enforced (longest-match
precedence, `Crawl-delay` honored), with per-domain budgets logged.

## Architecture

```
Next.js (Vercel)                Convex (reactive backend)
┌────────────────────┐          ┌─────────────────────────────────┐
│ app/feed           │  live    │ crons → pipeline.runPipeline     │
│ app/bookmarks      │ queries  │   ingest→normalize→enrich→dedup  │
│ app/settings       │◀────────▶│   →rank→summarize→persist        │
│ app/dashboard      │ mutations│ feed.getFeed (read-time ranking) │
│ app/api/*          │   REST   │ auth · feedback · bookmarks · eval│
└────────────────────┘          └─────────────────────────────────┘
        lib/pipeline/*  ← pure, reusable, stage-isolated TS (shared)
```

- **Pipeline** (`web/lib/pipeline/`) is pure TS, decoupled from the app — see [web/PIPELINE.md](./web/PIPELINE.md).
- **Backend** (`web/convex/`) orchestrates + persists; ranking runs read-time so prefs/weights apply instantly.
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
npx convex run sources:seed          # seed the source catalog + ranking config + gold set
npx convex run pipeline:runPipeline '{"trigger":"manual"}'   # pull the first batch
```

The pipeline then runs automatically every 20 minutes (`web/convex/crons.ts`).

## REST API

A read-only JSON API is exposed two ways: directly from Convex (`*.convex.site`)
and mirrored on the Vercel domain via Next.js route handlers (`web/app/api/*`) that
proxy the same Convex queries with ETag revalidation + pagination.

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/feed?topic=&limit=&offset=` | Globally-ranked representative items + `ETag` (send `If-None-Match` for `304`). |
| `GET` | `/api/cluster?id=` | A cluster's representative + related members. |
| `POST` | `/api/feedback` | Record `{itemId, action}`; requires `Authorization: Bearer <convex-token>`. |

```bash
curl -s "https://hud-news.vercel.app/api/feed?limit=2" | jq '.items[].title'
```

## Evaluation & MLOps

- **CI metric gates** (`npm run eval:gate`): Precision@10 / nDCG@10 / DupF1 are
  computed over a stratified **gold set** through the real ranking + dedup code;
  CI **fails** if any drops below its threshold (`web/lib/pipeline/evalMetrics.ts`).
- **Measured enrichment**: entity (NER) + topic precision/recall against a tiny
  labeled set (`evaluation.nerTopicEval`, `web/lib/pipeline/nerEval.ts`).
- **Auto-rollback**: when an eval run regresses beyond a threshold vs the
  previously-promoted config version, the live config is reverted and a critical
  alert is raised.
- **A/B canary**: a running experiment deterministically routes users (by hashed
  id) to a control/variant ranking-config version; the feed applies it read-time
  and per-arm satisfaction/lift is recorded each pipeline run.
- **Dense embeddings (BYO key)**: if `OPENAI_API_KEY` is set on the Convex
  deployment, item vectors use `text-embedding-3-small`; otherwise the pipeline
  falls back to the local hashing-trick vector (never a hard dependency).
- **Prompt registry** (`web/lib/pipeline/prompts.ts`): every LLM prompt is
  versioned, type-specific, and hot-swappable via an `ACTIVE` selector — no
  hardcoded prompts at call sites.
- **Cost tracking** (`web/lib/pipeline/cost.ts` + `mlops.costSummary`): a pricing
  table + token estimators give an estimated USD breakdown (embeddings + summaries)
  over a recent window for the ops console.
- **Healthcheck** (`npm run healthcheck`): post-deploy smoke test that hits the
  live `/api/feed` and fails on non-200, empty, or stale data
  (`web/scripts/healthcheck.mjs`, logic in `web/lib/health.ts`).
- **AI-agent rules** (`web/.claude/rules/`): `code-style.md` + `testing.md` capture
  the architecture/test conventions for AI coding agents.

## Environment variables

| Where | Var | Purpose |
|---|---|---|
| `web/.env.local` (Next.js) | `NEXT_PUBLIC_CONVEX_URL` | Convex client URL (written by `convex dev`); also used by the `/api/*` route handlers |
| Convex deployment | `KEY_ENCRYPTION_SECRET` | AES-GCM secret for BYO API keys |
| Convex deployment | `OPENAI_API_KEY` | _optional_ — enables dense embeddings in the pipeline |
| Convex deployment | `SITE_URL`, `JWT_PRIVATE_KEY`, `JWKS` | Convex Auth (set by `@convex-dev/auth`) |

## Deploy (Vercel + Convex)

CI/CD is `.github/workflows/hud-web.yml` (typecheck → unit tests → eval gate →
build → deploy Convex + Vercel on push to `main`). Manual:

```bash
cd web
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
│  └─ api/                    REST route handlers: /api/feed, /api/cluster, /api/feedback
├─ components/hud/            AppFrame, FeedView, NewsCard, Gauge, BreakingTicker, Settings/Bookmarks/Dashboard views
├─ lib/pipeline/              ingest · normalize · enrich · dedup · rank · summarize ·
│                             prompts (registry) · cost · evalMetrics · nerEval · experiment (pure TS)
├─ lib/                       api · convexHttp · health (route/script helpers)
├─ convex/                    schema, auth(+authz RBAC), pipeline orchestrator, crons, feed/feedback/bookmarks/eval/mlops/dashboard
├─ scripts/                   healthcheck.mjs (post-deploy smoke test)
├─ .claude/rules/             code-style.md · testing.md (AI-agent context)
└─ proxy.ts                   Next 16 auth middleware (route gating)
```

Built with Next.js, Convex, and Claude Code.
