import { query, mutation, internalMutation } from "./_generated/server";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";
import { DEFAULT_PREFS } from "./defaults";
import { Doc, Id } from "./_generated/dataModel";

function withDefaults(prefs: Doc<"userPrefs"> | null, userId: Id<"users">) {
  return {
    userId,
    focusTopics: prefs?.focusTopics ?? DEFAULT_PREFS.focusTopics,
    mutedSources: prefs?.mutedSources ?? DEFAULT_PREFS.mutedSources,
    boostedSources: prefs?.boostedSources ?? DEFAULT_PREFS.boostedSources,
    autoScrollSpeed: prefs?.autoScrollSpeed ?? DEFAULT_PREFS.autoScrollSpeed,
    focusVsPopularMix: prefs?.focusVsPopularMix ?? DEFAULT_PREFS.focusVsPopularMix,
    bookmarkResurfaceHours:
      prefs?.bookmarkResurfaceHours ?? DEFAULT_PREFS.bookmarkResurfaceHours,
    quietHours: prefs?.quietHours,
    onboarded: prefs?.onboarded ?? DEFAULT_PREFS.onboarded,
  };
}

/** Current user's prefs, filled with defaults if not yet saved. */
export const getPrefs = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return null;
    const prefs = await ctx.db
      .query("userPrefs")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .unique();
    return withDefaults(prefs, userId);
  },
});

export const updatePrefs = mutation({
  args: {
    focusTopics: v.optional(v.array(v.string())),
    mutedSources: v.optional(v.array(v.string())),
    boostedSources: v.optional(v.array(v.string())),
    autoScrollSpeed: v.optional(v.number()),
    focusVsPopularMix: v.optional(v.number()),
    bookmarkResurfaceHours: v.optional(v.number()),
    quietHours: v.optional(
      v.object({ start: v.number(), end: v.number(), timezoneOffset: v.optional(v.number()) }),
    ),
    onboarded: v.optional(v.boolean()),
  },
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const existing = await ctx.db
      .query("userPrefs")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .unique();
    const base = withDefaults(existing, userId);
    const next = { ...base, ...clean(args) };
    if (existing) {
      await ctx.db.patch(existing._id, next);
    } else {
      await ctx.db.insert("userPrefs", next);
    }
    return next;
  },
});

/** Used by feedback (mute source) to add to muted list. */
export const muteSource = internalMutation({
  args: { userId: v.id("users"), sourceId: v.string() },
  handler: async (ctx, { userId, sourceId }) => {
    const existing = await ctx.db
      .query("userPrefs")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .unique();
    const base = withDefaults(existing, userId);
    if (base.mutedSources.includes(sourceId)) return;
    const muted = [...base.mutedSources, sourceId];
    if (existing) await ctx.db.patch(existing._id, { mutedSources: muted });
    else await ctx.db.insert("userPrefs", { ...base, mutedSources: muted });
  },
});

function clean<T extends Record<string, unknown>>(obj: T): Partial<T> {
  const out: Partial<T> = {};
  for (const [k, val] of Object.entries(obj)) {
    if (val !== undefined) out[k as keyof T] = val as T[keyof T];
  }
  return out;
}
