import { query } from "./_generated/server";
import { getAuthUserId } from "@convex-dev/auth/server";
import { DEFAULT_CONFIG, DEFAULT_PREFS } from "./defaults";
import { topicalMatch } from "../lib/pipeline/rank";

const BREAKING_WINDOW_MS = 3 * 3600 * 1000;

/**
 * Notification decisioning: surface "breaking" events = high engagement OR
 * high cross-source velocity within the last few hours, biased toward the
 * user's interests. Collapsed to one ping per cluster, max 3.
 */
export const getBreaking = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return [];

    const cfg = await ctx.db
      .query("pipelineConfig")
      .withIndex("by_key", (q) => q.eq("key", "default"))
      .unique();
    const velThreshold = cfg?.breakingVelocityThreshold ?? DEFAULT_CONFIG.breakingVelocityThreshold;

    const prefs = await ctx.db
      .query("userPrefs")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .unique();
    const focusTopics = prefs?.focusTopics ?? DEFAULT_PREFS.focusTopics;

    const cutoff = Date.now() - BREAKING_WINDOW_MS;
    const recent = await ctx.db
      .query("items")
      .withIndex("by_publishedAt", (q) => q.gte("publishedAt", cutoff))
      .collect();

    const candidates = recent
      .filter((it) => it.isRepresentative)
      .map((it) => {
        const interest = topicalMatch(it.topics, focusTopics, it.sourceId, []);
        const hot = it.engagement.points >= 120 || it.features.velocity >= Math.min(1, velThreshold / 6);
        const breakingScore = it.features.popularity + it.features.velocity + 0.5 * interest;
        return { it, hot, interest, breakingScore };
      })
      .filter((c) => c.hot)
      .sort((a, b) => b.breakingScore - a.breakingScore)
      .slice(0, 3);

    return candidates.map((c) => ({
      _id: c.it._id,
      title: c.it.title,
      url: c.it.url,
      sourceName: c.it.sourceName,
      points: c.it.engagement.points,
      onFocus: c.interest >= 0.34,
    }));
  },
});
