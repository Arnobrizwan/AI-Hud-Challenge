import { query } from "./_generated/server";
import { getAuthUserId } from "@convex-dev/auth/server";

const WINDOW_MS = 48 * 3600 * 1000;

/** Operator overview: throughput, sources, clusters, feedback, distributions. */
export const overview = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return null;

    const cutoff = Date.now() - WINDOW_MS;
    const items = await ctx.db
      .query("items")
      .withIndex("by_publishedAt", (q) => q.gte("publishedAt", cutoff))
      .collect();
    const sources = await ctx.db.query("sources").collect();
    const clusters = await ctx.db
      .query("clusters")
      .withIndex("by_lastUpdated", (q) => q.gte("lastUpdatedAt", cutoff))
      .collect();

    // items per source (top)
    const perSource = new Map<string, number>();
    for (const it of items) perSource.set(it.sourceName, (perSource.get(it.sourceName) ?? 0) + 1);
    const topSources = Array.from(perSource.entries())
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    // topic distribution
    const perTopic = new Map<string, number>();
    for (const it of items) for (const t of it.topics) perTopic.set(t, (perTopic.get(t) ?? 0) + 1);
    const topicDist = Array.from(perTopic.entries())
      .map(([topic, count]) => ({ topic, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 12);

    // feedback breakdown (last 500)
    const fb = await ctx.db.query("feedback").order("desc").take(500);
    const fbCounts: Record<string, number> = {};
    for (const f of fb) fbCounts[f.action] = (fbCounts[f.action] ?? 0) + 1;

    const runs = await ctx.db
      .query("pipelineRuns")
      .withIndex("by_startedAt")
      .order("desc")
      .take(20);

    const multiSourceClusters = clusters.filter((c) => c.memberCount > 1).length;

    return {
      itemCount: items.length,
      clusterCount: clusters.length,
      multiSourceClusters,
      dedupeRate: items.length ? 1 - clusters.length / items.length : 0,
      sourcesEnabled: sources.filter((s) => s.enabled).length,
      sourcesTotal: sources.length,
      sourcesErrored: sources.filter((s) => (s.errorCount ?? 0) > 0 && !s.lastSuccessAt).length,
      topSources,
      topicDist,
      fbCounts,
      runs: runs.map((r) => ({
        _id: r._id,
        startedAt: r.startedAt,
        finishedAt: r.finishedAt ?? null,
        status: r.status,
        trigger: r.trigger,
        durationMs: r.finishedAt ? r.finishedAt - r.startedAt : null,
        counts: r.counts,
        stages: r.stages,
      })),
    };
  },
});
