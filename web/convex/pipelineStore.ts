import { internalMutation } from "./_generated/server";
import { v } from "convex/values";
import { Id, Doc } from "./_generated/dataModel";
import { hammingHex } from "../lib/pipeline/text";
import { computeTopics } from "../lib/pipeline/enrich";

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
  readableText: v.optional(v.string()),
  rawHtml: v.optional(v.string()),
  contentHash: v.optional(v.string()),
  entityLinks: v.optional(v.array(v.object({ name: v.string(), qid: v.string() }))),
  simhash: v.optional(v.string()),
  vector: v.optional(v.array(v.number())),
  flagged: v.optional(v.boolean()),
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
    const existingByKey = new Map<string, Doc<"items">>();
    for (const it of items) {
      const ex = await ctx.db
        .query("items")
        .withIndex("by_dedupeKey", (q) => q.eq("dedupeKey", it.dedupeKey))
        .unique();
      if (ex) existingByKey.set(it.dedupeKey, ex);
    }

    // recent items for INCREMENTAL clustering (attach new items to prior events
    // via SimHash Hamming distance — streaming dedup across batches).
    const recentForMerge = await ctx.db
      .query("items")
      .withIndex("by_publishedAt", (q) => q.gte("publishedAt", fetchedAt - 24 * 3600 * 1000))
      .take(600);

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

    // 2. insert new items; refresh hotter dups + trendlets + incremental merge
    let inserted = 0;
    let duplicates = 0;
    let updatedTrend = 0;
    let merged = 0;
    const repByCluster = new Map<number, Id<"items">>();
    for (const it of items) {
      const existing = existingByKey.get(it.dedupeKey);
      const clusterId = clusterIdByIndex.get(it.clusterIndex);
      if (existing) {
        duplicates++;
        const patch: Record<string, unknown> = {};
        if (it.engagement.points > existing.engagement.points) {
          patch.engagement = it.engagement;
          patch.features = it.features;
        }
        // trendlet: content changed since last fetch → "updated"
        if (it.contentHash && existing.contentHash && it.contentHash !== existing.contentHash) {
          patch.trendlet = "updated";
          patch.version = (existing.version ?? 1) + 1;
          patch.updatedAt = fetchedAt;
          patch.summaryExtractive = it.summaryExtractive;
          patch.contentHash = it.contentHash;
          updatedTrend++;
        }
        if (Object.keys(patch).length) await ctx.db.patch(existing._id, patch);
        if (it.isRepresentative && clusterId) repByCluster.set(it.clusterIndex, existing._id);
        continue;
      }

      // incremental clustering: near-dup of a recent item from a prior batch?
      let mergeClusterId: Id<"clusters"> | undefined;
      if (it.simhash) {
        for (const r of recentForMerge) {
          if (r.simhash && r.clusterId && hammingHex(it.simhash, r.simhash) <= 3) {
            mergeClusterId = r.clusterId;
            break;
          }
        }
      }

      const { clusterIndex, isRepresentative, ...rest } = it;
      const finalCluster = mergeClusterId ?? clusterId;
      const id = await ctx.db.insert("items", {
        ...rest,
        fetchedAt,
        clusterId: finalCluster,
        isRepresentative: mergeClusterId ? false : isRepresentative,
        trendlet: "new" as const,
        version: 1,
      });
      inserted++;
      if (mergeClusterId) {
        merged++;
        const c = await ctx.db.get(mergeClusterId);
        if (c) await ctx.db.patch(mergeClusterId, {
          memberCount: c.memberCount + 1,
          lastUpdatedAt: fetchedAt,
        });
      } else if (isRepresentative) {
        repByCluster.set(clusterIndex, id);
      }
    }

    // 3. set representative pointer on clusters
    for (const [idx, itemId] of repByCluster) {
      const cid = clusterIdByIndex.get(idx);
      if (cid) await ctx.db.patch(cid, { representativeItemId: itemId });
    }

    return { inserted, duplicates, clusters: clusterIdByIndex.size, merged, updatedTrend };
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

/**
 * Backfill: recompute topics for existing items with the current classifier
 * (e.g. after fixing aggregator source-topic flooding). Run once via the CLI:
 *   npx convex run pipelineStore:reclassifyTopics --prod
 */
export const reclassifyTopics = internalMutation({
  args: { limit: v.optional(v.number()) },
  handler: async (ctx, { limit }) => {
    const cutoff = Date.now() - 5 * 24 * 3600 * 1000;
    const items = await ctx.db
      .query("items")
      .withIndex("by_publishedAt", (q) => q.gte("publishedAt", cutoff))
      .take(limit ?? 2000);
    const sources = await ctx.db.query("sources").collect();
    const srcTopics = new Map(sources.map((s) => [s.sourceId, s.topics]));
    let updated = 0;
    for (const it of items) {
      const topics = computeTopics(
        it.title,
        it.readableText || it.summaryExtractive || "",
        srcTopics.get(it.sourceId) ?? [],
        it.kind,
      );
      if (JSON.stringify(topics) !== JSON.stringify(it.topics)) {
        await ctx.db.patch(it._id, { topics });
        updated++;
      }
    }
    return { scanned: items.length, updated };
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
