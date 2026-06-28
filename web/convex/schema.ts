import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";
import { authTables } from "@convex-dev/auth/server";

/**
 * HUD — High-Signal Personal News Feed
 * Convex data model. See web/PIPELINE.md for how each table feeds the pipeline.
 */
export default defineSchema({
  // Convex Auth identity tables (users, authSessions, authAccounts, ...)
  ...authTables,

  // ---- Per-user preferences -------------------------------------------------
  userPrefs: defineTable({
    userId: v.id("users"),
    focusTopics: v.array(v.string()), // e.g. ["ai", "startups"]
    mutedSources: v.array(v.string()), // source ids
    boostedSources: v.array(v.string()),
    autoScrollSpeed: v.number(), // px/sec, 0 = paused
    focusVsPopularMix: v.number(), // 0 = all popular, 1 = all focus
    bookmarkResurfaceHours: v.number(),
    quietHours: v.optional(v.object({ start: v.number(), end: v.number() })), // 0-23
    onboarded: v.boolean(),
  }).index("by_user", ["userId"]),

  // ---- BYO AI keys (ciphertext only) ---------------------------------------
  apiKeys: defineTable({
    userId: v.id("users"),
    provider: v.union(v.literal("openai"), v.literal("anthropic")),
    ciphertext: v.string(), // AES-GCM, encrypted with KEY_ENCRYPTION_SECRET
    last4: v.string(), // for display only
    model: v.optional(v.string()),
    valid: v.boolean(),
    updatedAt: v.number(),
  }).index("by_user", ["userId"])
    .index("by_user_provider", ["userId", "provider"]),

  // ---- Content sources ------------------------------------------------------
  sources: defineTable({
    sourceId: v.string(), // stable slug
    name: v.string(),
    kind: v.union(
      v.literal("rss"),
      v.literal("hackernews"),
      v.literal("reddit"),
      v.literal("x"),
      v.literal("newsletter"),
    ),
    url: v.string(), // feed url or handle
    topics: v.array(v.string()),
    weight: v.number(), // source reputation 0..1
    enabled: v.boolean(),
    // ingest bookkeeping
    etag: v.optional(v.string()),
    lastModified: v.optional(v.string()),
    lastFetchedAt: v.optional(v.number()),
    lastSuccessAt: v.optional(v.number()),
    errorCount: v.number(),
    successCount: v.number(),
    lastError: v.optional(v.string()),
  }).index("by_sourceId", ["sourceId"])
    .index("by_enabled", ["enabled"]),

  // ---- Normalized + enriched items -----------------------------------------
  items: defineTable({
    dedupeKey: v.string(), // hash(canonicalUrl|normalizedTitle)
    sourceId: v.string(),
    sourceName: v.string(),
    kind: v.string(),
    title: v.string(),
    url: v.string(),
    canonicalUrl: v.string(),
    summaryExtractive: v.string(),
    summaryAbstractive: v.optional(v.string()),
    image: v.optional(v.string()),
    author: v.optional(v.string()),
    lang: v.string(),
    publishedAt: v.number(),
    fetchedAt: v.number(),
    wordCount: v.number(),
    topics: v.array(v.string()),
    entities: v.array(v.string()),
    contentType: v.string(), // news | opinion | release | discussion
    // raw popularity engagement (as reported by the source)
    engagement: v.object({
      points: v.number(), // HN points / Reddit ups / X likes
      comments: v.number(),
    }),
    // computed signal features (transparent ranking inputs)
    features: v.object({
      recency: v.number(),
      sourceWeight: v.number(),
      popularity: v.number(), // normalized z-score 0..1
      velocity: v.number(),
    }),
    clusterId: v.optional(v.id("clusters")),
    isRepresentative: v.boolean(),
  }).index("by_dedupeKey", ["dedupeKey"])
    .index("by_publishedAt", ["publishedAt"])
    .index("by_cluster", ["clusterId"])
    .index("by_source", ["sourceId"]),

  // ---- Event clusters (dedup grouping) -------------------------------------
  clusters: defineTable({
    representativeItemId: v.optional(v.id("items")),
    title: v.string(),
    memberCount: v.number(),
    topics: v.array(v.string()),
    velocity: v.number(), // members added per hour
    firstSeenAt: v.number(),
    lastUpdatedAt: v.number(),
    popularity: v.number(),
  }).index("by_lastUpdated", ["lastUpdatedAt"]),

  // ---- Per user x item ranking scores --------------------------------------
  scores: defineTable({
    userId: v.id("users"),
    itemId: v.id("items"),
    clusterId: v.optional(v.id("clusters")),
    score: v.number(),
    lane: v.union(v.literal("focus"), v.literal("trending")),
    breakdown: v.object({
      recency: v.number(),
      sourceWeight: v.number(),
      topicalMatch: v.number(),
      novelty: v.number(),
      velocity: v.number(),
      popularity: v.number(),
    }),
    publishedAt: v.number(),
    computedAt: v.number(),
  }).index("by_user_score", ["userId", "score"])
    .index("by_user_item", ["userId", "itemId"]),

  // ---- Feedback signals -----------------------------------------------------
  feedback: defineTable({
    userId: v.id("users"),
    itemId: v.id("items"),
    action: v.union(
      v.literal("up"),
      v.literal("down"),
      v.literal("save"),
      v.literal("not_interested"),
      v.literal("mute_source"),
      v.literal("click"),
      v.literal("dwell"),
      v.literal("seen"),
    ),
    value: v.optional(v.number()), // dwell ms etc.
    createdAt: v.number(),
  }).index("by_user", ["userId"])
    .index("by_user_item", ["userId", "itemId"])
    .index("by_user_action", ["userId", "action"]),

  // ---- Bookmarks ------------------------------------------------------------
  bookmarks: defineTable({
    userId: v.id("users"),
    itemId: v.id("items"),
    note: v.optional(v.string()),
    savedAt: v.number(),
    lastResurfacedAt: v.optional(v.number()),
  }).index("by_user", ["userId"])
    .index("by_user_item", ["userId", "itemId"]),

  // ---- Pipeline run telemetry ----------------------------------------------
  pipelineRuns: defineTable({
    startedAt: v.number(),
    finishedAt: v.optional(v.number()),
    status: v.union(v.literal("running"), v.literal("ok"), v.literal("error")),
    trigger: v.string(), // cron | manual
    stages: v.array(v.object({
      name: v.string(),
      ms: v.number(),
      inCount: v.number(),
      outCount: v.number(),
      error: v.optional(v.string()),
    })),
    counts: v.object({
      fetched: v.number(),
      inserted: v.number(),
      duplicates: v.number(),
      clusters: v.number(),
    }),
    error: v.optional(v.string()),
  }).index("by_startedAt", ["startedAt"]),

  // ---- Evaluation runs (offline metrics) -----------------------------------
  evalRuns: defineTable({
    createdAt: v.number(),
    k: v.number(),
    metrics: v.object({
      precisionAtK: v.number(),
      ndcgAtK: v.number(),
      coverage: v.number(),
      novelty: v.number(),
      dupF1: v.number(),
      diversity: v.number(),
    }),
    sampleSize: v.number(),
    notes: v.optional(v.string()),
  }).index("by_createdAt", ["createdAt"]),

  // ---- Hot-reloadable pipeline config (singleton row) ----------------------
  pipelineConfig: defineTable({
    key: v.string(), // "default"
    weights: v.object({
      recency: v.number(),
      sourceWeight: v.number(),
      topicalMatch: v.number(),
      novelty: v.number(),
      velocity: v.number(),
      popularity: v.number(),
    }),
    recencyHalfLifeHours: v.number(),
    breakingVelocityThreshold: v.number(),
    explorationEpsilon: v.number(),
    maxPerSourcePerScreen: v.number(),
    updatedAt: v.number(),
  }).index("by_key", ["key"]),
});
