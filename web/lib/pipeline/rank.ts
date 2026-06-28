import type { EnrichedItem } from "./types";

/**
 * Stage 5 — ranking. Two halves, both pure:
 *   1. computeItemFeatures(): user-independent signals (recency, sourceWeight,
 *      popularity z-score across the batch, velocity). Stored on the item.
 *   2. scoreForUser(): personalized score = transparent weighted blend of the
 *      item features + topical match + novelty, biased by the focus↔popular mix.
 *
 * Every output keeps a `breakdown` so the HUD can show *why* a card surfaced.
 */

export interface RankWeights {
  recency: number;
  sourceWeight: number;
  topicalMatch: number;
  novelty: number;
  velocity: number;
  popularity: number;
}

export interface ItemFeatures {
  recency: number;
  sourceWeight: number;
  popularity: number;
  velocity: number;
}

export function recencyScore(publishedAt: number, halfLifeHours: number, now = Date.now()): number {
  const ageHours = Math.max(0, (now - publishedAt) / 3_600_000);
  return Math.pow(2, -ageHours / Math.max(0.5, halfLifeHours));
}

/** Compute batch-global item features. `velocityByItem` is members/hour. */
export function computeItemFeatures(
  items: EnrichedItem[],
  opts: { halfLifeHours: number; velocityByIndex?: number[]; now?: number },
): ItemFeatures[] {
  const now = opts.now ?? Date.now();
  // popularity: log-compressed engagement, min-max normalized within the batch,
  // then blended with a source-weight prior so zero-engagement feeds aren't 0.
  const rawPop = items.map((it) => Math.log1p((it.points ?? 0) + 2 * (it.comments ?? 0)));
  const maxPop = Math.max(0, ...rawPop);
  const velMax = Math.max(1, ...(opts.velocityByIndex ?? [1]));

  return items.map((it, i) => {
    const engagement = maxPop > 0 ? rawPop[i] / maxPop : 0;
    const popularity = 0.7 * engagement + 0.3 * it.sourceWeight;
    const velocity = opts.velocityByIndex
      ? Math.min(1, opts.velocityByIndex[i] / velMax)
      : 0;
    return {
      recency: recencyScore(it.publishedAt, opts.halfLifeHours, now),
      sourceWeight: it.sourceWeight,
      popularity: Math.min(1, popularity),
      velocity,
    };
  });
}

export interface UserContext {
  focusTopics: string[];
  boostedSources: string[];
  mutedSources: string[];
  focusVsPopularMix: number; // 0 popular .. 1 focus
  seen: Set<string>; // item ids the user has already been shown
}

export interface UserScore {
  score: number;
  lane: "focus" | "trending";
  breakdown: {
    recency: number;
    sourceWeight: number;
    topicalMatch: number;
    novelty: number;
    velocity: number;
    popularity: number;
  };
}

export function topicalMatch(
  itemTopics: string[],
  focusTopics: string[],
  sourceId: string,
  boosted: string[],
): number {
  if (focusTopics.length === 0) return 0.3;
  const focus = new Set(focusTopics);
  let overlap = 0;
  for (const t of itemTopics) if (focus.has(t)) overlap++;
  let match = Math.min(1, overlap / Math.min(3, focusTopics.length));
  if (boosted.includes(sourceId)) match = Math.min(1, match + 0.25);
  return match;
}

/** Stage 5b — personalized score for one item given user context + weights. */
export function scoreForUser(
  item: { topics: string[]; sourceId: string; id: string; features: ItemFeatures },
  ctx: UserContext,
  weights: RankWeights,
): UserScore {
  const tm = topicalMatch(item.topics, ctx.focusTopics, item.sourceId, ctx.boostedSources);
  const novelty = ctx.seen.has(item.id) ? 0.2 : 1;
  const f = item.features;

  const base = weights.recency * f.recency + weights.sourceWeight * f.sourceWeight;
  const focusComponent = weights.topicalMatch * tm + weights.novelty * novelty;
  const popularComponent = weights.popularity * f.popularity + weights.velocity * f.velocity;

  const mix = clamp01(ctx.focusVsPopularMix);
  const wFocus = 0.4 + 0.6 * mix;
  const wPop = 0.4 + 0.6 * (1 - mix);

  let score = base + wFocus * focusComponent + wPop * popularComponent;
  if (ctx.mutedSources.includes(item.sourceId)) score *= 0.001; // effectively hidden

  const lane: "focus" | "trending" =
    wFocus * focusComponent >= wPop * popularComponent ? "focus" : "trending";

  return {
    score,
    lane,
    breakdown: {
      recency: f.recency,
      sourceWeight: f.sourceWeight,
      topicalMatch: tm,
      novelty,
      velocity: f.velocity,
      popularity: f.popularity,
    },
  };
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}
