import { query, mutation, internalQuery, internalMutation } from "./_generated/server";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";
import { SEED_SOURCES } from "./seedData";
import { SEED_GOLD } from "./goldData";
import { DEFAULT_CONFIG } from "./defaults";

const kindValidator = v.union(
  v.literal("rss"),
  v.literal("hackernews"),
  v.literal("reddit"),
  v.literal("x"),
  v.literal("newsletter"),
  v.literal("jsonfeed"),
);

/** Public: list all sources with health stats (for dashboard + settings). */
export const listSources = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("sources").collect();
    return rows.sort((a, b) => b.weight - a.weight);
  },
});

export const toggleSource = mutation({
  args: { sourceId: v.string(), enabled: v.boolean() },
  handler: async (ctx, { sourceId, enabled }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const row = await ctx.db
      .query("sources")
      .withIndex("by_sourceId", (q) => q.eq("sourceId", sourceId))
      .unique();
    if (row) await ctx.db.patch(row._id, { enabled });
  },
});

export const upsertSource = mutation({
  args: {
    sourceId: v.string(),
    name: v.string(),
    kind: kindValidator,
    url: v.string(),
    topics: v.array(v.string()),
    weight: v.number(),
    enabled: v.boolean(),
  },
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const row = await ctx.db
      .query("sources")
      .withIndex("by_sourceId", (q) => q.eq("sourceId", args.sourceId))
      .unique();
    if (row) {
      await ctx.db.patch(row._id, args);
    } else {
      await ctx.db.insert("sources", {
        ...args,
        errorCount: 0,
        successCount: 0,
      });
    }
  },
});

/** Idempotent: seed the default source catalog + pipeline config if empty. */
export const seed = mutation({
  args: {},
  handler: async (ctx) => {
    const existing = await ctx.db.query("sources").collect();
    const byId = new Map(existing.map((s) => [s.sourceId, s]));
    let inserted = 0;
    let updated = 0;
    for (const s of SEED_SOURCES) {
      const cur = byId.get(s.sourceId);
      if (cur) {
        // Refresh catalog to the default definition (corrected URLs, enabled
        // state, weights). seed() is the authoritative "reset to defaults".
        if (
          cur.url !== s.url ||
          cur.name !== s.name ||
          cur.weight !== s.weight ||
          cur.enabled !== s.enabled
        ) {
          await ctx.db.patch(cur._id, {
            url: s.url,
            name: s.name,
            topics: s.topics,
            weight: s.weight,
            kind: s.kind,
            enabled: s.enabled,
          });
          updated++;
        }
        continue;
      }
      await ctx.db.insert("sources", { ...s, errorCount: 0, successCount: 0 });
      inserted++;
    }
    const cfg = await ctx.db
      .query("pipelineConfig")
      .withIndex("by_key", (q) => q.eq("key", "default"))
      .unique();
    if (!cfg) {
      await ctx.db.insert("pipelineConfig", {
        ...DEFAULT_CONFIG,
        updatedAt: Date.now(),
      });
    }
    // Seed the curated gold evaluation set once (idempotent).
    const goldExisting = await ctx.db.query("goldSet").take(1);
    let goldInserted = 0;
    if (goldExisting.length === 0) {
      for (const g of SEED_GOLD) {
        await ctx.db.insert("goldSet", { ...g, createdAt: Date.now() });
        goldInserted++;
      }
    }
    return { inserted, updated, total: existing.length + inserted, goldInserted };
  },
});

// ---- internal (pipeline) --------------------------------------------------

export const listEnabled = internalQuery({
  args: {},
  handler: async (ctx) => {
    return await ctx.db
      .query("sources")
      .withIndex("by_enabled", (q) => q.eq("enabled", true))
      .collect();
  },
});

export const recordFetch = internalMutation({
  args: {
    sourceId: v.string(),
    ok: v.boolean(),
    etag: v.optional(v.string()),
    lastModified: v.optional(v.string()),
    error: v.optional(v.string()),
  },
  handler: async (ctx, { sourceId, ok, etag, lastModified, error }) => {
    const row = await ctx.db
      .query("sources")
      .withIndex("by_sourceId", (q) => q.eq("sourceId", sourceId))
      .unique();
    if (!row) return;
    await ctx.db.patch(row._id, {
      lastFetchedAt: Date.now(),
      ...(ok
        ? {
            lastSuccessAt: Date.now(),
            successCount: row.successCount + 1,
            etag: etag ?? row.etag,
            lastModified: lastModified ?? row.lastModified,
            lastError: undefined,
          }
        : { errorCount: row.errorCount + 1, lastError: error }),
    });
  },
});
