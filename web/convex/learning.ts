import { internalMutation, query } from "./_generated/server";
import { Id } from "./_generated/dataModel";

/**
 * Learning-to-rank: aggregate implicit/explicit feedback into per-source
 * satisfaction priors (CTR, save-rate, mute-rate). Recomputed by the cron and
 * blended into ranking by `scoreForUser` (sourceSatisfaction).
 */
export const recomputeSourceStats = internalMutation({
  args: {},
  handler: async (ctx) => {
    const fb = await ctx.db.query("feedback").order("desc").take(2000);
    if (fb.length === 0) return { sources: 0 };

    // map itemId -> sourceId (dedup item loads)
    const itemIds = Array.from(new Set(fb.map((f) => f.itemId)));
    const itemSource = new Map<string, string>();
    for (const id of itemIds) {
      const it = await ctx.db.get(id as Id<"items">);
      if (it) itemSource.set(id, it.sourceId);
    }

    type Agg = { impressions: number; clicks: number; saves: number; mutes: number };
    const agg = new Map<string, Agg>();
    const bump = (s: string, k: keyof Agg) => {
      const a = agg.get(s) ?? { impressions: 0, clicks: 0, saves: 0, mutes: 0 };
      a[k]++;
      agg.set(s, a);
    };
    for (const f of fb) {
      const s = itemSource.get(f.itemId);
      if (!s) continue;
      if (f.action === "seen") bump(s, "impressions");
      else if (f.action === "click") bump(s, "clicks");
      else if (f.action === "up" || f.action === "more_like_this") bump(s, "saves");
      else if (f.action === "mute_source" || f.action === "down" || f.action === "not_interested")
        bump(s, "mutes");
    }

    let count = 0;
    for (const [sourceId, a] of agg) {
      const imp = Math.max(1, a.impressions + a.clicks + a.saves);
      const ctr = a.clicks / imp;
      const saveRate = a.saves / imp;
      const muteRate = a.mutes / imp;
      const satisfaction = Math.max(0, Math.min(1, 0.5 + 0.6 * (saveRate + ctr) - 0.8 * muteRate));
      const existing = await ctx.db
        .query("sourceStats")
        .withIndex("by_sourceId", (q) => q.eq("sourceId", sourceId))
        .unique();
      const doc = {
        sourceId,
        impressions: a.impressions,
        clicks: a.clicks,
        saves: a.saves,
        mutes: a.mutes,
        ctr,
        saveRate,
        muteRate,
        satisfaction,
        updatedAt: Date.now(),
      };
      if (existing) await ctx.db.patch(existing._id, doc);
      else await ctx.db.insert("sourceStats", doc);
      count++;
    }
    return { sources: count };
  },
});

/** Per-source satisfaction map for the ranker. */
export const sourceSatisfaction = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("sourceStats").collect();
    const out: Record<string, number> = {};
    for (const r of rows) out[r.sourceId] = r.satisfaction;
    return out;
  },
});
