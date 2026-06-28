import { query } from "./_generated/server";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";
import { DEFAULT_PREFS, DEFAULT_CONFIG } from "./defaults";
import { scoreForUser, type UserContext } from "../lib/pipeline/rank";
import { effectiveConfig } from "./config";
import { Doc } from "./_generated/dataModel";

// 72h so daily newsletters (TLDR AI, AI News, The Rundown) reliably appear
// alongside high-frequency sources like HackerNews.
const FEED_WINDOW_MS = 72 * 3600 * 1000;

/**
 * The HUD feed. Read-time personalized ranking over recent representative
 * items, so the stream is always fresh and reacts to pref/feedback changes.
 */
export const getFeed = query({
  args: { topic: v.optional(v.string()), limit: v.optional(v.number()) },
  handler: async (ctx, { topic, limit }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return { items: [], generatedAt: Date.now() };

    const prefsRow = await ctx.db
      .query("userPrefs")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .unique();
    // A/B canary: route this user to control/variant config (deterministic).
    const eff = await effectiveConfig(ctx, userId);
    const weights = eff.weights;
    const maxPerSource = eff.maxPerSourcePerScreen;
    const epsilon = eff.explorationEpsilon;

    // learned per-source satisfaction prior (learning-to-rank)
    const statRows = await ctx.db.query("sourceStats").collect();
    const sourceSatisfaction: Record<string, number> = {};
    for (const s of statRows) sourceSatisfaction[s.sourceId] = s.satisfaction;

    const cutoff = Date.now() - FEED_WINDOW_MS;
    const recent = await ctx.db
      .query("items")
      .withIndex("by_publishedAt", (q) => q.gte("publishedAt", cutoff))
      .collect();
    const items = recent.filter((i) => i.isRepresentative);

    // user signals
    const fb = await ctx.db
      .query("feedback")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .collect();
    const seen = new Set<string>();
    const notInterested = new Set<string>();
    const downed = new Set<string>();
    for (const f of fb) {
      if (["seen", "click", "up", "down"].includes(f.action)) seen.add(f.itemId);
      if (f.action === "not_interested") notInterested.add(f.itemId);
      if (f.action === "down") downed.add(f.itemId);
    }
    const bookmarks = await ctx.db
      .query("bookmarks")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .collect();
    const bookmarked = new Set(bookmarks.map((b) => b.itemId));

    // cluster sizes for "+N related"
    const clusters = await ctx.db
      .query("clusters")
      .withIndex("by_lastUpdated", (q) => q.gte("lastUpdatedAt", cutoff))
      .collect();
    const clusterSize = new Map(clusters.map((c) => [c._id, c.memberCount]));

    const userCtx: UserContext = {
      focusTopics: prefsRow?.focusTopics ?? DEFAULT_PREFS.focusTopics,
      boostedSources: prefsRow?.boostedSources ?? DEFAULT_PREFS.boostedSources,
      mutedSources: prefsRow?.mutedSources ?? DEFAULT_PREFS.mutedSources,
      focusVsPopularMix: prefsRow?.focusVsPopularMix ?? DEFAULT_PREFS.focusVsPopularMix,
      seen,
      epsilon,
      sourceSatisfaction,
    };

    type Card = ReturnType<typeof shape>;
    const shape = (it: Doc<"items">) => {
      const s = scoreForUser(
        { topics: it.topics, sourceId: it.sourceId, id: it._id, features: it.features },
        userCtx,
        weights,
      );
      let score = s.score;
      if (downed.has(it._id)) score *= 0.25;
      if (it.flagged) score *= 0.05; // safety: down-rank flagged content
      return {
        _id: it._id,
        title: it.title,
        url: it.url,
        sourceId: it.sourceId,
        sourceName: it.sourceName,
        kind: it.kind,
        summary: it.summaryAbstractive ?? it.summaryExtractive,
        hasAbstractive: !!it.summaryAbstractive,
        image: it.image ?? null,
        publishedAt: it.publishedAt,
        topics: it.topics,
        entities: it.entities,
        contentType: it.contentType,
        engagement: it.engagement,
        score,
        lane: s.lane,
        breakdown: s.breakdown,
        bookmarked: bookmarked.has(it._id),
        related: it.clusterId ? Math.max(0, (clusterSize.get(it.clusterId) ?? 1) - 1) : 0,
        trendlet: it.trendlet ?? null,
        flagged: it.flagged ?? false,
        entityLinks: it.entityLinks ?? [],
      };
    };

    let cards: Card[] = items
      .filter((it) => !notInterested.has(it._id))
      .filter((it) => (topic && topic !== "all" ? it.topics.includes(topic) : true))
      .map(shape)
      .sort((a, b) => b.score - a.score);

    // per-source diversity cap (no single-source domination on screen)
    const perSource = new Map<string, number>();
    cards = cards.filter((c) => {
      const n = perSource.get(c.sourceId) ?? 0;
      if (n >= maxPerSource) return false;
      perSource.set(c.sourceId, n + 1);
      return true;
    });

    return {
      items: cards.slice(0, limit ?? 60),
      generatedAt: Date.now(),
      // surface the A/B arm so the console can attribute impressions + lift.
      experiment: eff.experiment ? { name: eff.experiment, arm: eff.arm, version: eff.version } : null,
    };
  },
});

/**
 * Public (REST) feed — global ranking, no per-user personalization. Backs the
 * documented `GET /api/feed` contract. Returns a stable list + an ETag basis.
 */
export const publicFeed = query({
  args: { topic: v.optional(v.string()), limit: v.optional(v.number()), offset: v.optional(v.number()) },
  handler: async (ctx, { topic, limit, offset }) => {
    const cutoff = Date.now() - FEED_WINDOW_MS;
    const cfgRow = await ctx.db
      .query("pipelineConfig")
      .withIndex("by_key", (q) => q.eq("key", "default"))
      .unique();
    const weights = cfgRow?.weights ?? DEFAULT_CONFIG.weights;

    const recent = await ctx.db
      .query("items")
      .withIndex("by_publishedAt", (q) => q.gte("publishedAt", cutoff))
      .collect();
    const items = recent
      .filter((i) => i.isRepresentative && !i.flagged)
      .filter((i) => (topic && topic !== "all" ? i.topics.includes(topic) : true));

    const ranked = items
      .map((it) => {
        const s = scoreForUser(
          { topics: it.topics, sourceId: it.sourceId, id: it._id, features: it.features },
          { focusTopics: [], boostedSources: [], mutedSources: [], focusVsPopularMix: 0.5, seen: new Set() },
          weights,
        );
        return { it, score: s.score };
      })
      .sort((a, b) => b.score - a.score);

    const off = offset ?? 0;
    const page = ranked.slice(off, off + (limit ?? 30));
    const latest = items.reduce((m, i) => Math.max(m, i.fetchedAt), 0);
    return {
      etag: `W/"${latest}-${ranked.length}"`,
      total: ranked.length,
      items: page.map(({ it, score }) => ({
        id: it._id,
        title: it.title,
        url: it.url,
        canonicalUrl: it.canonicalUrl,
        source: it.sourceName,
        kind: it.kind,
        summary: it.summaryAbstractive ?? it.summaryExtractive,
        publishedAt: it.publishedAt,
        topics: it.topics,
        clusterId: it.clusterId ?? null,
        engagement: it.engagement,
        score,
      })),
    };
  },
});

/** Lightweight feed stats for the HUD header readout. */
export const getFeedStats = query({
  args: {},
  handler: async (ctx) => {
    const cutoff = Date.now() - FEED_WINDOW_MS;
    const recent = await ctx.db
      .query("items")
      .withIndex("by_publishedAt", (q) => q.gte("publishedAt", cutoff))
      .collect();
    const sources = new Set(recent.map((r) => r.sourceId));
    const lastRun = await ctx.db
      .query("pipelineRuns")
      .withIndex("by_startedAt")
      .order("desc")
      .first();
    return {
      itemCount: recent.length,
      sourceCount: sources.size,
      lastRunAt: lastRun?.finishedAt ?? lastRun?.startedAt ?? null,
      lastRunStatus: lastRun?.status ?? null,
    };
  },
});
