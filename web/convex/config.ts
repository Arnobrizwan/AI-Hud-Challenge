import { query, mutation, internalQuery } from "./_generated/server";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";
import { DEFAULT_CONFIG } from "./defaults";

/** Hot-reloadable pipeline config (singleton). Returns defaults if unset. */
export const getConfig = query({
  args: {},
  handler: async (ctx) => {
    const row = await ctx.db
      .query("pipelineConfig")
      .withIndex("by_key", (q) => q.eq("key", "default"))
      .unique();
    if (!row) return { ...DEFAULT_CONFIG, updatedAt: 0 };
    return row;
  },
});

/** Internal variant used by the pipeline (no auth needed). */
export const getConfigInternal = internalQuery({
  args: {},
  handler: async (ctx) => {
    const row = await ctx.db
      .query("pipelineConfig")
      .withIndex("by_key", (q) => q.eq("key", "default"))
      .unique();
    if (!row) return { ...DEFAULT_CONFIG, updatedAt: 0 };
    return row;
  },
});

export const updateConfig = mutation({
  args: {
    weights: v.optional(
      v.object({
        recency: v.number(),
        sourceWeight: v.number(),
        topicalMatch: v.number(),
        novelty: v.number(),
        velocity: v.number(),
        popularity: v.number(),
      }),
    ),
    recencyHalfLifeHours: v.optional(v.number()),
    breakingVelocityThreshold: v.optional(v.number()),
    explorationEpsilon: v.optional(v.number()),
    maxPerSourcePerScreen: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    // Require an authenticated operator to change ranking config.
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");

    const existing = await ctx.db
      .query("pipelineConfig")
      .withIndex("by_key", (q) => q.eq("key", "default"))
      .unique();
    const merged = {
      key: "default",
      weights: args.weights ?? existing?.weights ?? DEFAULT_CONFIG.weights,
      recencyHalfLifeHours:
        args.recencyHalfLifeHours ??
        existing?.recencyHalfLifeHours ??
        DEFAULT_CONFIG.recencyHalfLifeHours,
      breakingVelocityThreshold:
        args.breakingVelocityThreshold ??
        existing?.breakingVelocityThreshold ??
        DEFAULT_CONFIG.breakingVelocityThreshold,
      explorationEpsilon:
        args.explorationEpsilon ??
        existing?.explorationEpsilon ??
        DEFAULT_CONFIG.explorationEpsilon,
      maxPerSourcePerScreen:
        args.maxPerSourcePerScreen ??
        existing?.maxPerSourcePerScreen ??
        DEFAULT_CONFIG.maxPerSourcePerScreen,
      updatedAt: Date.now(),
    };
    if (existing) await ctx.db.patch(existing._id, merged);
    else await ctx.db.insert("pipelineConfig", merged);
    return merged;
  },
});
