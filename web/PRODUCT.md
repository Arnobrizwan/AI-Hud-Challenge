# HUD — Product Document (six-pager)

**Product:** HUD — a high-signal personal news feed rendered as a heads-up
display. **Status:** v1 live (https://hud-news.vercel.app). **Owner:** Arnob
Rizwan. This doc is self-contained: an engineer can build HUD from it alone.

---

## 1. Problem, vision, and pain points

**Problem.** Knowledge workers (esp. in AI/tech) drown in feeds. The signal they
care about is scattered across HackerNews, AI newsletters, subreddits, and X, and
buried under duplicates, clickbait, and recency noise. Existing readers are either
*chronological* (no ranking) or *opaque black boxes* (Twitter/algorithmic) that
optimize engagement, not the user's stated focus.

**Vision.** A glanceable HUD that continuously surfaces the **right** items: a
blend of *your declared focus* and *what's genuinely breaking*, deduplicated into
events, ranked transparently, and streamed hands-free — with full user control
over the algorithm and your own AI keys.

**Pain points addressed.**
- *Information overload* → dedup into events + per-source diversity cap.
- *Recency tyranny* → recency is one weighted signal, not the only sort.
- *Black-box ranking* → every card shows **why** it surfaced; weights are user-editable.
- *Missing the important stuff* → breaking detection (velocity × interest) + bookmarks that resurface.
- *Vendor lock-in of AI* → bring-your-own key; app fully works without one.
- *Doom-scrolling* → auto-scroll with a pace **you** set; pause/keyboard control; "not interested"/mute.

**Non-goals (v1).** Social graph / following people; native mobile app; paid
content / paywalled full-text; human editorial curation; multi-language UI.

---

## 2. Users & user stories

**Primary persona — "Ravi", staff AI engineer.** Wants a 2-minute morning scan +
ambient awareness during the day. Cares about LLMs/agents, skeptical of hype, will
tune knobs.

**Secondary — "Mira", founder/operator.** Wants breaking + funding/market signal,
low effort, sane defaults, no tuning.

**Tertiary — "Sam", the pipeline operator (internal).** Tunes ranking, watches
quality/drift, manages sources.

**User stories (acceptance in parentheses).**
- As a reader I set focus topics so the feed prioritizes them. *(Focus chips in Settings change rank within one feed read.)*
- As a reader I slide focus↔trending to bias discovery. *(Mix slider visibly retags cards Focus/Trending.)*
- As a reader I control auto-scroll speed and pause. *(Slider + hover/`space` pause; `J/K` step.)*
- As a reader I understand why a card is shown. *("Why this?" reveals recency/topical/popularity/velocity/novelty/source bars.)*
- As a reader I save items and have them resurface later. *(Bookmark → reappears tagged "From bookmarks" after my set hours.)*
- As a reader I see one card per event, not 6 duplicates. *(Syndicated copies collapse with "+N related".)*
- As a reader I plug in my OpenAI/Anthropic key for better summaries. *(Encrypted; "test key"; extractive fallback if absent.)*
- As a reader I delete all my data. *(GDPR delete removes account + data.)*
- As an operator I tune ranking weights live. *(Dashboard sliders hot-reload; no redeploy.)*
- As an operator I roll back a bad config. *(Config registry → one-click rollback.)*
- As an operator I see quality/drift alerts. *(Alerts panel; drift sparkline.)*

---

## 3. Feature set

**Shipped (v1).** Auto-scrolling ranked feed; focus×trending mix; transparent
per-card scoring + gauges; MinHash/LSH + SimHash event dedup (incremental);
breaking ticker (cooldown, quiet-hours, per-topic thresholds); bookmarks +
periodic resurfacing; feedback (👍/👎/save/mute/not-interested/more-like-this);
BYO AI keys (AES-GCM) with grounded abstractive summaries; multi-user auth
(password + guest); operator dashboard (sources, hot-reload weights, eval gauges,
analytics, alerts, drift, config registry/rollback, A/B canary, labeling); REST
contract (`/api/feed`, `/api/cluster`, `/api/feedback`) + WebSub; trendlets
(NEW/UPDATED); Wikidata entity linking; content-safety flag; GDPR delete.

**Backlog / ideas (prioritized).**
1. **P0 — Reddit OAuth + X via paid API/bridge** (close source gaps reliably).
2. **P0 — Real embeddings** (BYO key) → semantic dedup + "related" + topic model upgrade.
3. **P1 — Digest/email + push notifications** (Web Push) for breaking.
4. **P1 — Saved searches / custom lanes** (e.g., "evals", "open models").
5. **P1 — Multi-device sync + reading position memory.**
6. **P2 — Team/shared HUDs** (org feed, shared boosts/mutes).
7. **P2 — Learned ranker (LTR model)** trained on the labeled set, replacing the linear blend.
8. **P2 — Native mobile / PWA install + offline cache.**
9. **P3 — Audio mode** (TTS digest), calendar-aware quiet hours, browser extension.

---

## 4. System design (build this)

**Stack.** Next.js 16 (App Router, TS, Tailwind v4) on Vercel; Convex (reactive DB
+ actions + cron + auth). One Vercel-native deploy; no separate backend.

```
sources ─▶ ingest ─▶ normalize ─▶ enrich ─▶ dedup ─▶ rank ─▶ summarize ─▶ persist
   (cron, 20m / WebSub push)         (pure TS stages, lib/pipeline/*)        │
                                                                       items+clusters
HUD (app/feed) ◀── reactive Convex queries (read-time personalized ranking) ─┘
feedback/bookmarks ─▶ learning (source satisfaction) ─▶ ranker prior
eval · drift · data-quality · alerts · config-registry  (ops, cron + dashboard)
```

**Modularity rule (critical).** Pipeline stages are **pure functions** in
`lib/pipeline/*` with typed contracts (`types.ts`), decoupled from Convex/UI. App
changes never force pipeline changes; swap a stage's internals without touching
callers. This is the AI-engineer requirement.

**Data model (Convex).** `items` (normalized+enriched+features+simhash+vector),
`clusters` (events), `sources`, `userPrefs`, `apiKeys` (ciphertext), `bookmarks`,
`feedback`, `scores`, `evalRuns`, `pipelineRuns`, `pipelineConfig`+`configVersions`,
`sourceStats`, `alerts`, `driftSnapshots`, `experiments`, `labels`, `subscriptions`,
`notificationsLog`, `goldSet`. Every user-scoped read/write gated by `getAuthUserId`.

**Key build sequence for a new engineer.** (1) `create-next-app` + Convex + Convex
Auth. (2) Schema + seed sources. (3) `lib/pipeline/*` pure stages + cron
orchestrator. (4) Read-time `feed.getFeed`. (5) HUD feed UI. (6) bookmarks/feedback/
notifications. (7) dashboard. (8) eval + learning + ops. See `README.md`,
`PIPELINE.md`, `FEATURE_DICTIONARY.md`, `RUNBOOK.md`.

---

## 5. Key design decisions (the challenge's questions)

**Auto-scroll speed.** Stored per-user (`autoScrollSpeed`, px/sec, default 26).
Driven by `requestAnimationFrame` (px = speed·dt), seamless loop via a duplicated
list; pauses on hover/focus; `space` toggles, `J/K` step ±140px. Default tuned so
an average headline+teaser is readable in one pass; users adjust live.

**Popularity metric.** `popularity = 0.7·minmax(log1p(points + 2·comments)) +
0.3·sourceWeight`, plus **velocity** (cross-source cluster growth). Engagement =
HN points / Reddit ups / X likes; comments weighted 2× (discussion = signal); log
compresses virality; min-max normalizes within the batch; the source-weight prior
keeps zero-engagement feeds (newsletters) from flatlining.

**Focus × other mix.** A single slider `focusVsPopularMix∈[0,1]`. Score =
`base + wFocus·(topical+novelty) + wPop·(popularity+velocity)` with
`wFocus=0.4+0.6·mix`, `wPop=0.4+0.6·(1−mix)` — both lanes always present, biased by
the slider. Each card is tagged **Focus** or **Trending** by which component
dominated. Per-source diversity cap prevents single-source domination.

**Bookmark management.** Toggle save → `bookmarks`; Saved page lists them; remove;
optional note. Saving also emits a positive feedback signal (improves ranking).

**Bookmark resurfacing frequency.** Per-user `bookmarkResurfaceHours` (default 48,
slider 6–168h). A saved item not re-shown within the window is re-injected into the
stream tagged "From bookmarks"; `lastResurfacedAt` tracked to avoid spamming. Why
user-set: resurfacing cadence is a personal habit (daily reviewer vs weekly).

**Ranking transparency.** Linear, inspectable weighted sum; weights live in
`pipelineConfig` and are editable from the dashboard with hot-reload. Future: an LTR
model trained on `labels`/feedback, kept behind the same `scoreForUser` interface.

---

## 6. Roadmap, metrics, risks

**Roadmap.**
- **Now (v1, done):** core feed + pipeline + dashboard + BYO keys + ops, deployed.
- **M1 (2–3 wks):** Reddit OAuth + X source reliability; Web Push breaking alerts; email digest; embeddings-based semantic dedup/related.
- **M2 (1–2 mo):** saved searches/custom lanes; reading-position sync; LTR model from labeled data; PWA install.
- **M3 (quarter):** team/shared HUDs; browser extension; audio digest; SLA/observability hardening (pager, cost dashboards).

**Success metrics (track from day 1).**
- *Quality:* Precision@10, nDCG@10, cluster purity, dup-F1, summary factuality.
- *User:* save rate, CTR, mute/not-interested rate, novelty, time-to-surface for breaking, D1/D7 retention, session length (capped — anti-doomscroll).
- *System:* ingest→rank P95 latency, pipeline error rate, cost/story.
- *Ops:* alert MTTR, rollback success.

**Risks & mitigations.** Source access (Reddit/X rate-limit datacenter IPs) →
OAuth/bridges, graceful degradation. LLM cost/hallucination → BYO keys + grounding
checks + extractive fallback. Filter bubble → ε-greedy exploration + diversity cap
+ novelty. Privacy → store metadata only, AES-GCM keys, GDPR delete, 5-day TTL.
Engagement-optimization trap → optimize *stated focus + breaking*, expose session
length as a guardrail metric, never a growth target.

**Open questions for next iteration.** Best default mix for new users? Cold-start
onboarding (pick topics vs infer from first clicks)? When to graduate from linear
ranker to LTR (data threshold)? Notification aggressiveness defaults?
