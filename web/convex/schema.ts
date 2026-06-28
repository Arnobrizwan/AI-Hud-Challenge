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
    // 0-23 local hours; timezoneOffset = minutes from UTC (e.g. -300 for EST)
    quietHours: v.optional(
      v.object({ start: v.number(), end: v.number(), timezoneOffset: v.optional(v.number()) }),
    ),
    // per-topic breaking-alert thresholds (0..1 interest needed to notify)
    topicThresholds: v.optional(v.array(v.object({ topic: v.string(), threshold: v.number() }))),
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
      v.literal("jsonfeed"),
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
    // ---- quality / abuse signals (auto-managed) ----
    spamScore: v.optional(v.number()), // 0..1, raises -> auto-downgrade weight
    qualityScore: v.optional(v.number()), // learned engagement quality 0..1
    autoDowngraded: v.optional(v.boolean()),
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
    // ---- extended pipeline signals (waves) ----
    simhash: v.optional(v.string()), // 64-bit SimHash (hex) for borderline dedup
    vector: v.optional(v.array(v.number())), // hashing-trick semantic vector
    contentHash: v.optional(v.string()), // for update-diff trendlets
    version: v.optional(v.number()), // bumped when content changes
    updatedAt: v.optional(v.number()),
    trendlet: v.optional(v.union(v.literal("new"), v.literal("updated"))),
    entityLinks: v.optional(
      v.array(v.object({ name: v.string(), qid: v.string() })),
    ), // Wikidata QIDs
    readableText: v.optional(v.string()), // readability-extracted main text
    rawHtml: v.optional(v.string()), // original source HTML, alongside readableText
    flagged: v.optional(v.boolean()), // NSFW/spam/profanity
  }).index("by_dedupeKey", ["dedupeKey"])
    .index("by_publishedAt", ["publishedAt"])
    .index("by_cluster", ["clusterId"])
    .index("by_source", ["sourceId"])
    .index("by_simhash", ["simhash"]),

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
      v.literal("more_like_this"),
    ),
    value: v.optional(v.number()), // dwell ms etc.
    createdAt: v.number(),
  }).index("by_user", ["userId"])
    .index("by_user_item", ["userId", "itemId"])
    .index("by_user_action", ["userId", "action"])
    .index("by_item", ["itemId"]),

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
      clusterPurity: v.optional(v.number()),
      factuality: v.optional(v.number()),
      timeToSurfaceMs: v.optional(v.number()),
    }),
    sampleSize: v.number(),
    notes: v.optional(v.string()),
    configVersion: v.optional(v.number()),
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
    version: v.optional(v.number()),
  }).index("by_key", ["key"]),

  // ---- Notification audit log (cooldown + dedup + escalation) ---------------
  notificationsLog: defineTable({
    userId: v.id("users"),
    itemId: v.id("items"),
    clusterId: v.optional(v.id("clusters")),
    sentAt: v.number(),
    reason: v.string(), // velocity | engagement | interest
    score: v.number(),
  }).index("by_user", ["userId"])
    .index("by_user_cluster", ["userId", "clusterId"]),

  // ---- Learned per-source stats (learning-to-rank prior) -------------------
  sourceStats: defineTable({
    sourceId: v.string(),
    impressions: v.number(),
    clicks: v.number(),
    saves: v.number(),
    mutes: v.number(),
    ctr: v.number(),
    saveRate: v.number(),
    muteRate: v.number(),
    satisfaction: v.number(), // 0..1 learned quality
    updatedAt: v.number(),
  }).index("by_sourceId", ["sourceId"]),

  // ---- Config registry: versioned ranking config + promote/rollback --------
  configVersions: defineTable({
    version: v.number(),
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
    createdAt: v.number(),
    note: v.optional(v.string()),
    promoted: v.boolean(), // was/is the active production config
  }).index("by_version", ["version"]),

  // ---- A/B experiments (canary on ranking config) --------------------------
  experiments: defineTable({
    name: v.string(),
    status: v.union(v.literal("running"), v.literal("stopped")),
    controlVersion: v.number(),
    variantVersion: v.number(),
    trafficPct: v.number(), // % of users on variant
    createdAt: v.number(),
    metrics: v.optional(v.object({
      controlSatisfaction: v.number(),
      variantSatisfaction: v.number(),
      samples: v.number(),
    })),
  }).index("by_status", ["status"]),

  // ---- Observability alerts -------------------------------------------------
  alerts: defineTable({
    type: v.string(), // source_outage | pipeline_error | drift | cost | data_quality
    severity: v.union(v.literal("info"), v.literal("warn"), v.literal("critical")),
    message: v.string(),
    createdAt: v.number(),
    resolved: v.boolean(),
  }).index("by_createdAt", ["createdAt"])
    .index("by_resolved", ["resolved"]),

  // ---- Drift snapshots (topic/entity distribution over time) ----------------
  driftSnapshots: defineTable({
    createdAt: v.number(),
    topicDist: v.array(v.object({ topic: v.string(), share: v.number() })),
    divergence: v.number(), // JS divergence vs previous window
  }).index("by_createdAt", ["createdAt"]),

  // ---- Human labels (internal labeling UI) ----------------------------------
  labels: defineTable({
    userId: v.id("users"),
    kind: v.union(
      v.literal("dup_pair"),
      v.literal("cluster_correct"),
      v.literal("summary_factual"),
    ),
    itemId: v.optional(v.id("items")),
    otherItemId: v.optional(v.id("items")),
    label: v.string(), // yes | no | unsure
    createdAt: v.number(),
  }).index("by_kind", ["kind"]),

  // ---- WebSub subscriptions -------------------------------------------------
  subscriptions: defineTable({
    sourceId: v.string(),
    topicUrl: v.string(),
    hubUrl: v.string(),
    secret: v.string(),
    status: v.union(
      v.literal("pending"),
      v.literal("active"),
      v.literal("failed"),
    ),
    leaseSeconds: v.optional(v.number()),
    subscribedAt: v.optional(v.number()),
    lastPingAt: v.optional(v.number()),
  }).index("by_sourceId", ["sourceId"]),

  // ---- Gold evaluation set (curated relevance labels) -----------------------
  goldSet: defineTable({
    topic: v.string(),
    keyword: v.string(), // a title substring marking a relevant story
    relevance: v.number(), // graded 0..1
    createdAt: v.number(),
  }).index("by_topic", ["topic"]),
});
