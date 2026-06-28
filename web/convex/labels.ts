import { mutation, query } from "./_generated/server";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";

/**
 * Human-in-the-loop labeling (section 9). Operators label dup-pairs, cluster
 * correctness, and summary factuality; labels accumulate into a training set.
 */
export const submit = mutation({
  args: {
    kind: v.union(v.literal("dup_pair"), v.literal("cluster_correct"), v.literal("summary_factual")),
    itemId: v.optional(v.id("items")),
    otherItemId: v.optional(v.id("items")),
    label: v.string(),
  },
  handler: async (ctx, a) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    await ctx.db.insert("labels", { ...a, userId, createdAt: Date.now() });
  },
});

/** A pair of recent same-titled items to label (dup-pair task). */
export const nextDupPair = query({
  args: {},
  handler: async (ctx) => {
    const cutoff = Date.now() - 48 * 3600 * 1000;
    const items = await ctx.db
      .query("items")
      .withIndex("by_publishedAt", (q) => q.gte("publishedAt", cutoff))
      .take(400);
    // find two items in the same cluster (candidate dup) or with similar titles
    for (const a of items) {
      if (!a.clusterId) continue;
      const b = items.find((x) => x._id !== a._id && x.clusterId === a.clusterId);
      if (b) {
        return {
          a: { id: a._id, title: a.title, source: a.sourceName },
          b: { id: b._id, title: b.title, source: b.sourceName },
        };
      }
    }
    return null;
  },
});

/** Exported labels = training dataset (section 9 deliverable). */
export const trainingSet = query({
  args: {},
  handler: async (ctx) => {
    const labels = await ctx.db.query("labels").order("desc").take(500);
    const counts: Record<string, number> = {};
    for (const l of labels) counts[l.kind] = (counts[l.kind] ?? 0) + 1;
    return { total: labels.length, counts, recent: labels.slice(0, 50) };
  },
});
