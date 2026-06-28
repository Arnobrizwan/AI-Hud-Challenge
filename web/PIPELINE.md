# HUD — AI/ML Pipeline Reference

How every stage of the high-signal news pipeline is implemented. The pipeline is
**modular and decoupled from the app**: each stage is a pure function in
`lib/pipeline/` with a typed input/output contract (`lib/pipeline/types.ts`), so
UI/schema changes don't ripple into the pipeline. Convex actions (`convex/`)
orchestrate the stages and persist results; the Next.js app only *reads* ranked
output reactively.

```
sources ─▶ ingest ─▶ normalize ─▶ enrich ─▶ dedup ─▶ rank ─▶ summarize ─▶ persist
                                                    │                         │
                                              (cluster events)         items + clusters
                                                                              │
                                  feed query (read-time personalization) ◀────┘
                                                                              │
                                                  feedback ─▶ eval ─▶ config (hot-reload)
```

Orchestrator: `convex/pipeline.ts` (`runPipeline`), scheduled every 20 min by
`convex/crons.ts` and triggerable from the dashboard (`triggerRun`).

---

## 0. Foundations & contracts
- **Types**: `lib/pipeline/types.ts` — `RawItem → NormalizedItem → EnrichedItem → ScoredItem`, plus `SourceSpec`.
- **Config**: hot-reloadable ranking weights + thresholds in the `pipelineConfig` table (`convex/config.ts`); edited live from the dashboard, read by the pipeline and the feed without redeploy.
- **Taxonomy**: 16-topic taxonomy in `convex/defaults.ts`.

## 1. Ingestion & normalization
- **Adapters** (`lib/pipeline/ingest/`):
  - `rss.ts` — generic RSS 2.0 / Atom parser (via `fast-xml-parser`, runs in Convex's V8 action runtime). Handles RSS, **Reddit**, **X/nitter**, and **newsletters** (TLDR AI, AI News/smol.ai, The Rundown). Sends a real User-Agent and **conditional GET** (`If-None-Match` / `If-Modified-Since`) — 304s short-circuit.
  - `hackernews.ts` — HN Algolia API (`front_page`) for real **points + comment counts** (the popularity signal).
- **Per-source health**: ETag/Last-Modified, success/error counts, last error are recorded back to the `sources` table (`sources.recordFetch`) and shown on the dashboard.
- **Normalize** (`lib/pipeline/normalize.ts`): canonical URL (strips `utm_*`/tracking, lowercases host, sorts params), `dedupeKey = hash(canonicalUrl|title)`, language guess, timezone-safe dates, extractive teaser, word count. Exact dups dropped within a batch.

## 2. Content extraction & cleanup
- `lib/pipeline/text.ts`: HTML stripping + entity decoding, tokenization (stopword-filtered), title normalization, extractive summary (sentence-boundary aware), and a stable non-crypto `hashString` (used for dedupe keys + MinHash).

## 3. Enrichment (entities, topics, signals)
- `lib/pipeline/enrich.ts`:
  - **Topics**: rules+keywords classifier over the 16-topic taxonomy (multi-label, source topics as a floor). Swappable for an LLM classifier without changing the stage signature.
  - **Entities**: known-org dictionary + capitalized proper-noun heuristic.
  - **Content type**: `news | opinion | release | discussion` (discussion for HN/Reddit/X).

## 4. Deduplication & event grouping
- `lib/pipeline/dedup.ts` (concept ported from the Python repo's `deduplication-service`):
  - **MinHash** signatures (48 permutations) over title+lead word-shingles (k=3).
  - **LSH banding** (12 bands) to find candidate near-dup pairs cheaply.
  - Confirm by estimated **Jaccard ≥ 0.5**, then **union-find** into event clusters.
  - **Canonical representative** = highest source weight, then earliest timestamp.
  - One representative per event is shown in the feed; members power "+N related".

## 5. Ranking (transparent, two-stage)
- `lib/pipeline/rank.ts`:
  - **Item features** (`computeItemFeatures`, user-independent, stored on the item): `recency` (half-life decay), `sourceWeight`, `popularity` (log-compressed engagement min-max normalized across the batch, blended with a source-weight prior so zero-engagement feeds aren't 0), `velocity` (cross-source cluster growth).
  - **User score** (`scoreForUser`, read-time in `convex/feed.ts`): transparent weighted blend of `recency · sourceWeight · topicalMatch · novelty · velocity · popularity`, **biased by the focus↔trending mix slider**. Muted sources suppressed, boosted sources lifted. Every card carries its full `breakdown` (shown under "Why this?").
  - **Popularity metric** = normalized engagement (HN points / Reddit ups / X likes) + cross-source velocity.
  - **Diversity**: per-source cap per screen (`maxPerSourcePerScreen`) so one source can't dominate.

## 6. Summarization & headline generation
- `lib/pipeline/summarize.ts`:
  - **Extractive** teaser always (zero-cost, no key).
  - **Abstractive** (optional, BYO key): OpenAI/Anthropic via `fetch`, with guardrails — source-grounding system prompt, length control, no-speculation, and a post-hoc **entity-grounding check** (`isGrounded`) that rejects summaries inventing names. Falls back to extractive on any failure. Generated **on demand per item** with the user's key (`items.enhanceSummary`).

## 7. Personalization
- Read-time in `convex/feed.ts`: explicit prefs (focus topics, mute/boost), implicit signals (up/down/click/seen → novelty), cold-start = global-popular + topic diversity. The mix slider trades focus vs. trending continuously.

## 8. Notification decisioning
- `convex/notifications.ts` (`getBreaking`): breaking = high engagement OR high cross-source velocity in the last 3h, biased by interest, collapsed to one ping per event (max 3). Surfaced as the HUD "Breaking" ticker.

## 9. Feedback & human-in-the-loop
- `convex/feedback.ts`: up / down / not-interested / mute-source / click / seen. `mute_source` updates prefs; signals feed novelty and the eval relevance labels.

## 10. Evaluation suite
- `convex/evaluation.ts` (`runEval`): **Precision@K, nDCG@K, coverage, novelty, diversity**, plus a **dedupe-quality (Dup F1) proxy** from title-collision clustering. Relevance = explicit feedback when present, else a topical-match proxy. Results stored in `evalRuns` and rendered as gauges on the dashboard.

## 11. MLOps & orchestration
- `convex/pipeline.ts` records a `pipelineRuns` row per run with **per-stage timing, in/out counts, and errors**; housekeeping prunes items older than 3 days. The cron (`convex/crons.ts`) drives the schedule. Config changes hot-reload (no redeploy).

## 12. Drift, abuse & safety
- Language filter (English-only into ranking), per-source error tracking with auto-surfaced failures, mute/blocklists per user, and summary grounding checks. Source weights act as a reputation prior.

## 13. Storage & indexing
- Convex document tables with explicit indexes (`convex/schema.ts`): `items` (by publishedAt / dedupeKey / cluster / source), `clusters`, `scores`, `feedback`, `bookmarks`, `evalRuns`, `pipelineRuns`, `pipelineConfig`, `sources`, plus Convex Auth tables. BYO keys stored as **AES-GCM ciphertext only** (`convex/crypto.ts`).

## 14. Real-time interface to the HUD
- The HUD reads via **reactive Convex queries** (`feed.getFeed`, `feed.getFeedStats`, `notifications.getBreaking`, `dashboard.overview`, `bookmarks.*`) — live updates with no polling. Ranking is computed read-time so pref/weight changes reflect instantly.

## 15. Observability
- Dashboard (`/dashboard`): throughput, cluster/dedupe stats, source health, ranking-weight editor, eval gauges, feedback breakdown, and recent-run timings.

## 16. How to extend
- **Add a source**: insert a row in `sources` (dashboard "Sources") or extend `convex/seedData.ts`; pick a `kind` (`rss`/`hackernews`/`reddit`/`x`/`newsletter`). RSS-shaped sources need no code.
- **Add a pipeline stage**: add a pure function in `lib/pipeline/`, call it from `convex/pipeline.ts`. Contracts in `types.ts` keep it isolated.
- **Swap a model**: topic classifier and summarizer are behind stage functions — replace internals without touching callers.
