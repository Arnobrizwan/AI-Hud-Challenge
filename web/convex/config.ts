import { query, mutation, internalQuery, type QueryCtx } from "./_generated/server";
import { v } from "convex/values";
import { DEFAULT_CONFIG } from "./defaults";
import { requireAdmin } from "./authz";
import { assignArm } from "../lib/pipeline/experiment";

/**
 * Resolve the ranking config a given user should see, applying any RUNNING A/B
 * canary: the user is deterministically routed to the control or variant config
 * version (configVersions table) by `assignArm`. Falls back to the live default
 * config when no experiment is running or the version row is missing. Shared by
 * feed.ts and evaluation.ts so traffic routing is consistent everywhere.
 */
export async function effectiveConfig(ctx: QueryCtx, userId: string | null) {
  const cfgRow = await ctx.db
    .query("pipelineConfig")
    .withIndex("by_key", (q) => q.eq("key", "default"))
    .unique();
  const out = {
    weights: cfgRow?.weights ?? DEFAULT_CONFIG.weights,
    recencyHalfLifeHours: cfgRow?.recencyHalfLifeHours ?? DEFAULT_CONFIG.recencyHalfLifeHours,
    explorationEpsilon: cfgRow?.explorationEpsilon ?? DEFAULT_CONFIG.explorationEpsilon,
    maxPerSourcePerScreen: cfgRow?.maxPerSourcePerScreen ?? DEFAULT_CONFIG.maxPerSourcePerScreen,
    version: cfgRow?.version,
    arm: null as "control" | "variant" | null,
    experiment: null as string | null,
  };
  if (!userId) return out;

  const exp = await ctx.db
    .query("experiments")
    .withIndex("by_status", (q) => q.eq("status", "running"))
    .first();
  if (!exp) return out;

  const { arm, version } = assignArm(userId, exp);
  out.arm = arm;
  out.experiment = exp.name;
  const ver = await ctx.db
    .query("configVersions")
    .withIndex("by_version", (q) => q.eq("version", version))
    .unique();
  if (ver) {
    out.weights = ver.weights;
    out.recencyHalfLifeHours = ver.recencyHalfLifeHours;
    out.explorationEpsilon = ver.explorationEpsilon;
    out.maxPerSourcePerScreen = ver.maxPerSourcePerScreen;
    out.version = ver.version;
  }
  return out;
}

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
    // Ranking config is global — admins only.
    await requireAdmin(ctx);

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
