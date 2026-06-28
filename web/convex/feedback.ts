import { mutation } from "./_generated/server";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";
import { DEFAULT_PREFS } from "./defaults";

const actionValidator = v.union(
  v.literal("up"),
  v.literal("down"),
  v.literal("not_interested"),
  v.literal("mute_source"),
  v.literal("click"),
  v.literal("dwell"),
  v.literal("seen"),
  v.literal("more_like_this"),
);

/** Record a feedback signal; mute_source also updates prefs. */
export const record = mutation({
  args: { itemId: v.id("items"), action: actionValidator, value: v.optional(v.number()) },
  handler: async (ctx, { itemId, action, value }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    await ctx.db.insert("feedback", {
      userId,
      itemId,
      action,
      value,
      createdAt: Date.now(),
    });

    if (action === "mute_source") {
      const item = await ctx.db.get(itemId);
      if (item) {
        const prefs = await ctx.db
          .query("userPrefs")
          .withIndex("by_user", (q) => q.eq("userId", userId))
          .unique();
        const muted = new Set(prefs?.mutedSources ?? DEFAULT_PREFS.mutedSources);
        muted.add(item.sourceId);
        if (prefs) {
          await ctx.db.patch(prefs._id, { mutedSources: Array.from(muted) });
        } else {
          await ctx.db.insert("userPrefs", {
            userId,
            ...DEFAULT_PREFS,
            mutedSources: Array.from(muted),
          });
        }
      }
    }
  },
});

/** Batch "seen" marker — fired as cards scroll past, to drive novelty. */
export const markSeen = mutation({
  args: { itemIds: v.array(v.id("items")) },
  handler: async (ctx, { itemIds }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return;
    // de-dup against already-seen to keep the table lean
    const existing = await ctx.db
      .query("feedback")
      .withIndex("by_user_action", (q) => q.eq("userId", userId).eq("action", "seen"))
      .collect();
    const have = new Set(existing.map((e) => e.itemId));
    for (const itemId of itemIds) {
      if (have.has(itemId)) continue;
      await ctx.db.insert("feedback", {
        userId,
        itemId,
        action: "seen",
        createdAt: Date.now(),
      });
    }
  },
});
