import { query, action, internalQuery, internalMutation } from "./_generated/server";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";
import { internal } from "./_generated/api";
import { decryptSecret } from "./crypto";
import { abstractiveSummary, type Provider } from "../lib/pipeline/summarize";

/** Item detail + related cluster members. */
export const getItem = query({
  args: { itemId: v.id("items") },
  handler: async (ctx, { itemId }) => {
    const item = await ctx.db.get(itemId);
    if (!item) return null;
    let related: { _id: string; title: string; url: string; sourceName: string }[] = [];
    if (item.clusterId) {
      const members = await ctx.db
        .query("items")
        .withIndex("by_cluster", (q) => q.eq("clusterId", item.clusterId))
        .collect();
      related = members
        .filter((m) => m._id !== item._id)
        .map((m) => ({ _id: m._id, title: m.title, url: m.url, sourceName: m.sourceName }));
    }
    return { item, related };
  },
});

/** Public (REST) cluster view: representative + related members. */
export const publicCluster = query({
  args: { clusterId: v.id("clusters") },
  handler: async (ctx, { clusterId }) => {
    const cluster = await ctx.db.get(clusterId);
    if (!cluster) return null;
    const members = await ctx.db
      .query("items")
      .withIndex("by_cluster", (q) => q.eq("clusterId", clusterId))
      .collect();
    return {
      id: clusterId,
      title: cluster.title,
      memberCount: cluster.memberCount,
      topics: cluster.topics,
      velocity: cluster.velocity,
      members: members.map((m) => ({
        id: m._id, title: m.title, url: m.url, source: m.sourceName,
        publishedAt: m.publishedAt, isRepresentative: m.isRepresentative,
      })),
    };
  },
});

export const getRawForSummary = internalQuery({
  args: { itemId: v.id("items") },
  handler: async (ctx, { itemId }) => {
    const item = await ctx.db.get(itemId);
    if (!item) return null;
    return {
      title: item.title,
      text: item.summaryExtractive,
      hasAbstractive: !!item.summaryAbstractive,
    };
  },
});

export const setAbstractive = internalMutation({
  args: { itemId: v.id("items"), summary: v.string() },
  handler: async (ctx, { itemId, summary }) => {
    await ctx.db.patch(itemId, { summaryAbstractive: summary });
  },
});

/**
 * Generate an abstractive summary for an item using the signed-in user's BYO
 * key. No-op (returns null) if the user has no valid key — the UI then keeps
 * the extractive teaser.
 */
export const enhanceSummary = action({
  args: { itemId: v.id("items") },
  handler: async (ctx, { itemId }): Promise<{ summary: string | null }> => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    const raw = await ctx.runQuery(internal.items.getRawForSummary, { itemId });
    if (!raw) return { summary: null };

    for (const provider of ["openai", "anthropic"] as Provider[]) {
      const cipher = await ctx.runQuery(internal.apiKeys.getCipher, { userId, provider });
      if (!cipher) continue;
      const key = await decryptSecret(cipher.ciphertext);
      const summary = await abstractiveSummary({
        provider,
        key,
        model: cipher.model,
        title: raw.title,
        text: raw.text,
      });
      if (summary) {
        await ctx.runMutation(internal.items.setAbstractive, { itemId, summary });
        return { summary };
      }
    }
    return { summary: null };
  },
});
