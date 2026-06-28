import { query, mutation } from "./_generated/server";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";
import { DEFAULT_CONFIG, DEFAULT_PREFS } from "./defaults";
import { topicalMatch } from "../lib/pipeline/rank";

const COOLDOWN_MS = 3 * 3600 * 1000; // one alert per cluster per 3h
function inQuietHours(q: { start: number; end: number } | undefined, hour: number): boolean {
  if (!q) return false;
  return q.start <= q.end ? hour >= q.start && hour < q.end : hour >= q.start || hour < q.end;
}

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
    const topicThresholds = new Map(
      (prefs?.topicThresholds ?? []).map((t) => [t.topic, t.threshold]),
    );

    // quiet hours / DND: suppress all breaking pings.
    const hour = new Date(Date.now()).getUTCHours();
    if (inQuietHours(prefs?.quietHours, hour)) return [];

    // cooldown / cross-cluster dedup: skip clusters alerted in the last window.
    const recentLog = await ctx.db
      .query("notificationsLog")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .order("desc")
      .take(50);
    const onCooldown = new Set(
      recentLog
        .filter((l) => Date.now() - l.sentAt < COOLDOWN_MS && l.clusterId)
        .map((l) => String(l.clusterId)),
    );

    const cutoff = Date.now() - BREAKING_WINDOW_MS;
    const recent = await ctx.db
      .query("items")
      .withIndex("by_publishedAt", (q) => q.gte("publishedAt", cutoff))
      .collect();

    const candidates = recent
      .filter((it) => it.isRepresentative)
      .filter((it) => !it.flagged && !(it.clusterId && onCooldown.has(String(it.clusterId))))
      .map((it) => {
        const interest = topicalMatch(it.topics, focusTopics, it.sourceId, []);
        // per-topic alert threshold (default 0): require interest >= threshold
        const thr = Math.max(...it.topics.map((t) => topicThresholds.get(t) ?? 0), 0);
        const hot =
          interest >= thr &&
          (it.engagement.points >= 120 || it.features.velocity >= Math.min(1, velThreshold / 6));
        const breakingScore = it.features.popularity + it.features.velocity + 0.5 * interest;
        return { it, hot, interest, breakingScore };
      })
      .filter((c) => c.hot)
      .sort((a, b) => b.breakingScore - a.breakingScore)
      .slice(0, 3);

    return candidates.map((c) => ({
      _id: c.it._id,
      clusterId: c.it.clusterId ?? null,
      title: c.it.title,
      url: c.it.url,
      sourceName: c.it.sourceName,
      points: c.it.engagement.points,
      onFocus: c.interest >= 0.34,
    }));
  },
});

/** Audit log: record that a breaking ping was shown (drives cooldown). */
export const markShown = mutation({
  args: { itemId: v.id("items"), clusterId: v.optional(v.id("clusters")), reason: v.string(), score: v.number() },
  handler: async (ctx, a) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return;
    await ctx.db.insert("notificationsLog", { userId, ...a, sentAt: Date.now() });
  },
});
