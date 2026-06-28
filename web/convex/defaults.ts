/**
 * Shared defaults + taxonomy. Imported by Convex functions and the pipeline.
 * Plain data only (no Convex function exports) so it is safe to import anywhere.
 */

export const TOPIC_TAXONOMY = [
  "ai",
  "llm",
  "ml",
  "agents",
  "startups",
  "programming",
  "open-source",
  "security",
  "crypto",
  "science",
  "hardware",
  "robotics",
  "data",
  "business",
  "design",
  "policy",
] as const;

export type Topic = (typeof TOPIC_TAXONOMY)[number];

export const DEFAULT_PREFS = {
  focusTopics: ["ai", "llm", "agents", "startups"],
  mutedSources: [] as string[],
  boostedSources: [] as string[],
  autoScrollSpeed: 26, // px / second
  focusVsPopularMix: 0.6, // 0 popular .. 1 focus
  bookmarkResurfaceHours: 48,
  onboarded: false,
};

export const DEFAULT_CONFIG = {
  key: "default",
  weights: {
    recency: 0.22,
    sourceWeight: 0.12,
    topicalMatch: 0.28,
    novelty: 0.1,
    velocity: 0.12,
    popularity: 0.16,
  },
  recencyHalfLifeHours: 8,
  breakingVelocityThreshold: 5,
  explorationEpsilon: 0.12,
  maxPerSourcePerScreen: 3,
};
