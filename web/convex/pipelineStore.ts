import { internalMutation } from "./_generated/server";
import { v } from "convex/values";
import { Id } from "./_generated/dataModel";

/**
 * Persistence for the pipeline. Kept separate from the orchestrator action so
 * these run in the Convex transactional (query/mutation) runtime.
 */

const itemInput = v.object({
  dedupeKey: v.string(),
  sourceId: v.string(),
  sourceName: v.string(),
  kind: v.string(),
  title: v.string(),
  url: v.string(),
  canonicalUrl: v.string(),
  summaryExtractive: v.string(),
  image: v.optional(v.string()),
  author: v.optional(v.string()),
  lang: v.string(),
  publishedAt: v.number(),
  wordCount: v.number(),
  topics: v.array(v.string()),
  entities: v.array(v.string()),
  contentType: v.string(),
  engagement: v.object({ points: v.number(), comments: v.number() }),
  features: v.object({
    recency: v.number(),
    sourceWeight: v.number(),
    popularity: v.number(),
    velocity: v.number(),
  }),
  clusterIndex: v.number(),
  isRepresentative: v.boolean(),
});

const clusterInput = v.object({
  title: v.string(),
  memberCount: v.number(),
  topics: v.array(v.string()),
  velocity: v.number(),
  popularity: v.number(),
});

export const storeResults = internalMutation({
  args: {
    fetchedAt: v.number(),
    clusters: v.array(clusterInput),
    items: v.array(itemInput),
  },
  handler: async (ctx, { fetchedAt, clusters, items }) => {
    // 1. resolve existing copies up front (exact-dup by dedupeKey)
    const existingByKey = new Map<string, Id<"items">>();
    const existingPoints = new Map<string, number>();
    for (const it of items) {
      const ex = await ctx.db
        .query("items")
        .withIndex("by_dedupeKey", (q) => q.eq("dedupeKey", it.dedupeKey))
        .unique();
      if (ex) {
        existingByKey.set(it.dedupeKey, ex._id);
        existingPoints.set(it.dedupeKey, ex.engagement.points);
      }
    }

    // Only materialize clusters that contain at least one NEW item — otherwise
    // every all-duplicate rerun would leak empty cluster docs forever.
    const clusterHasNew = new Set<number>();
    for (const it of items) if (!existingByKey.has(it.dedupeKey)) clusterHasNew.add(it.clusterIndex);

    const clusterIdByIndex = new Map<number, Id<"clusters">>();
    for (let idx = 0; idx < clusters.length; idx++) {
      if (!clusterHasNew.has(idx)) continue;
      const c = clusters[idx];
      const id = await ctx.db.insert("clusters", {
        representativeItemId: undefined,
        title: c.title,
        memberCount: c.memberCount,
        topics: c.topics,
        velocity: c.velocity,
        firstSeenAt: fetchedAt,
        lastUpdatedAt: fetchedAt,
        popularity: c.popularity,
      });
      clusterIdByIndex.set(idx, id);
    }

    // 2. insert new items; refresh hotter dups
    let inserted = 0;
    let duplicates = 0;
    const repByCluster = new Map<number, Id<"items">>();
    for (const it of items) {
      const existingId = existingByKey.get(it.dedupeKey);
      const clusterId = clusterIdByIndex.get(it.clusterIndex);
      if (existingId) {
        duplicates++;
        if (it.engagement.points > (existingPoints.get(it.dedupeKey) ?? 0)) {
          await ctx.db.patch(existingId, { engagement: it.engagement, features: it.features });
        }
        if (it.isRepresentative && clusterId) repByCluster.set(it.clusterIndex, existingId);
        continue;
      }
      const { clusterIndex, isRepresentative, ...rest } = it;
      const id = await ctx.db.insert("items", { ...rest, fetchedAt, clusterId, isRepresentative });
      inserted++;
      if (isRepresentative) repByCluster.set(clusterIndex, id);
    }

    // 3. set representative pointer on clusters
    for (const [idx, itemId] of repByCluster) {
      const cid = clusterIdByIndex.get(idx);
      if (cid) await ctx.db.patch(cid, { representativeItemId: itemId });
    }

    return { inserted, duplicates, clusters: clusterIdByIndex.size };
  },
});

// ---- pipeline run telemetry ----------------------------------------------

export const startRun = internalMutation({
  args: { trigger: v.string() },
  handler: async (ctx, { trigger }) => {
    return await ctx.db.insert("pipelineRuns", {
      startedAt: Date.now(),
      status: "running",
      trigger,
      stages: [],
      counts: { fetched: 0, inserted: 0, duplicates: 0, clusters: 0 },
    });
  },
});

export const finishRun = internalMutation({
  args: {
    runId: v.id("pipelineRuns"),
    status: v.union(v.literal("ok"), v.literal("error")),
    stages: v.array(
      v.object({
        name: v.string(),
        ms: v.number(),
        inCount: v.number(),
        outCount: v.number(),
        error: v.optional(v.string()),
      }),
    ),
    counts: v.object({
      fetched: v.number(),
      inserted: v.number(),
      duplicates: v.number(),
      clusters: v.number(),
    }),
    error: v.optional(v.string()),
  },
  handler: async (ctx, { runId, ...rest }) => {
    await ctx.db.patch(runId, { ...rest, finishedAt: Date.now() });
  },
});

/** Housekeeping: drop items + clusters older than `olderThanMs` to bound storage. */
export const pruneOld = internalMutation({
  args: { olderThanMs: v.number() },
  handler: async (ctx, { olderThanMs }) => {
    const cutoff = Date.now() - olderThanMs;
    const oldItems = await ctx.db
      .query("items")
      .withIndex("by_publishedAt", (q) => q.lt("publishedAt", cutoff))
      .take(400);
    for (const row of oldItems) await ctx.db.delete(row._id);

    const oldClusters = await ctx.db
      .query("clusters")
      .withIndex("by_lastUpdated", (q) => q.lt("lastUpdatedAt", cutoff))
      .take(600);
    for (const row of oldClusters) await ctx.db.delete(row._id);

    return { items: oldItems.length, clusters: oldClusters.length };
  },
});
