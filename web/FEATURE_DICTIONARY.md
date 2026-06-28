# Feature Dictionary

Every ranking/pipeline feature: calculation, range, and where it's computed.
(Spec section 16 deliverable.)

## Ranking features (`lib/pipeline/rank.ts`)

| Feature | Calc | Range | Computed |
|---|---|---|---|
| `recency` | `2^(-ageHours / halfLife)` | 0..1 | `computeItemFeatures` |
| `sourceWeight` | source reputation, blended 0.6·static + 0.4·learned satisfaction | 0..1 | item + `scoreForUser` |
| `popularity` | `0.7·minmax(log1p(points + 2·comments)) + 0.3·sourceWeight` | 0..1 | `computeItemFeatures` |
| `velocity` | cross-source cluster growth (members/window), normalized | 0..1 | pipeline (cluster size) |
| `topicalMatch` | `min(1, overlap(itemTopics, focusTopics)/min(3,#focus))` (+0.25 boosted) | 0..1 | `scoreForUser` |
| `novelty` | seen → 0.2, else 1.0 | {0.2,1} | `scoreForUser` (per-user `seen`) |
| `satisfaction` | per-source `0.5 + 0.6·(saveRate+ctr) − 0.8·muteRate` | 0..1 | `learning.recomputeSourceStats` |
| `explorationBonus` | `ε · jitter(itemId)` for unseen items (ε-greedy) | 0..ε | `scoreForUser` |

**Final score** = `recency·w + sourceWeight·w + mixFocus·(topical·w + novelty·w) + mixPop·(popularity·w + velocity·w) + explorationBonus`, where `mixFocus = 0.4 + 0.6·mix`, `mixPop = 0.4 + 0.6·(1−mix)`. Muted source → ×0.001; flagged → ×0.05.

## Item signals

| Field | Calc | Computed |
|---|---|---|
| `simhash` | 64-bit SimHash over title+lead tokens (hex) | `text.simHash` |
| `vector` | 64-dim hashing-trick vector, L2-normalized | `text.hashingVector` |
| `contentHash` | hash(normTitle + lead) → drives NEW/UPDATED trendlets | `normalize` |
| `entityLinks` | Wikidata QIDs for representative entities | `kb.linkEntities` |
| `readableText` | readability main-content extraction | `text.readabilityExtract` |
| `flagged` | profanity/NSFW/spam keyword guard | `safety.isFlagged` |
| `trendlet` | `new` on insert, `updated` when contentHash changes | `pipelineStore` |

## Config (hot-reload, `pipelineConfig`)

`weights.{recency,sourceWeight,topicalMatch,novelty,velocity,popularity}` (each 0..0.5), `recencyHalfLifeHours`, `breakingVelocityThreshold`, `explorationEpsilon` (0..1), `maxPerSourcePerScreen`, `version`. Snapshots → `configVersions` (promote/rollback).

## Eval metrics (`convex/evaluation.ts`)

Precision@K, nDCG@K, coverage, novelty, diversity, dup-F1, clusterPurity, factuality, timeToSurfaceMs — all 0..1 except time (ms). Relevance = explicit feedback else topical-match proxy; gold-set hits if `goldSet` populated.
