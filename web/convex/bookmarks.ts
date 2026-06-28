import { query, mutation } from "./_generated/server";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";
import { DEFAULT_PREFS } from "./defaults";

/** Toggle a bookmark on/off; returns the new state. */
export const toggle = mutation({
  args: { itemId: v.id("items") },
  handler: async (ctx, { itemId }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const existing = await ctx.db
      .query("bookmarks")
      .withIndex("by_user_item", (q) => q.eq("userId", userId).eq("itemId", itemId))
      .unique();
    if (existing) {
      await ctx.db.delete(existing._id);
      return { bookmarked: false };
    }
    await ctx.db.insert("bookmarks", { userId, itemId, savedAt: Date.now() });
    await ctx.db.insert("feedback", {
      userId,
      itemId,
      action: "up",
      createdAt: Date.now(),
    });
    return { bookmarked: true };
  },
});

export const remove = mutation({
  args: { itemId: v.id("items") },
  handler: async (ctx, { itemId }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const existing = await ctx.db
      .query("bookmarks")
      .withIndex("by_user_item", (q) => q.eq("userId", userId).eq("itemId", itemId))
      .unique();
    if (existing) await ctx.db.delete(existing._id);
  },
});

/** Saved items, newest first, joined with their article. */
export const list = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return [];
    const rows = await ctx.db
      .query("bookmarks")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .order("desc")
      .collect();
    const out = [];
    for (const b of rows) {
      const item = await ctx.db.get(b.itemId);
      if (!item) continue;
      out.push({
        bookmarkId: b._id,
        savedAt: b.savedAt,
        note: b.note ?? null,
        _id: item._id,
        title: item.title,
        url: item.url,
        sourceName: item.sourceName,
        sourceId: item.sourceId,
        kind: item.kind,
        summary: item.summaryAbstractive ?? item.summaryExtractive,
        image: item.image ?? null,
        publishedAt: item.publishedAt,
        topics: item.topics,
        engagement: item.engagement,
      });
    }
    return out;
  },
});

/**
 * Bookmarks due to resurface: saved items not shown again within the user's
 * `bookmarkResurfaceHours` window. The feed injects these tagged FROM BOOKMARKS.
 */
export const resurfacing = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return [];
    const prefs = await ctx.db
      .query("userPrefs")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .unique();
    const hours = prefs?.bookmarkResurfaceHours ?? DEFAULT_PREFS.bookmarkResurfaceHours;
    const windowMs = hours * 3600 * 1000;
    const now = Date.now();

    const rows = await ctx.db
      .query("bookmarks")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .collect();
    const due = rows.filter((b) => {
      const last = b.lastResurfacedAt ?? b.savedAt;
      return now - last >= windowMs;
    });

    const out = [];
    for (const b of due.slice(0, 5)) {
      const item = await ctx.db.get(b.itemId);
      if (!item) continue;
      out.push({
        bookmarkId: b._id,
        _id: item._id,
        title: item.title,
        url: item.url,
        sourceName: item.sourceName,
        kind: item.kind,
        summary: item.summaryAbstractive ?? item.summaryExtractive,
        topics: item.topics,
      });
    }
    return out;
  },
});

export const markResurfaced = mutation({
  args: { bookmarkId: v.id("bookmarks") },
  handler: async (ctx, { bookmarkId }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const b = await ctx.db.get(bookmarkId);
    if (b && b.userId === userId) {
      await ctx.db.patch(bookmarkId, { lastResurfacedAt: Date.now() });
    }
  },
});
