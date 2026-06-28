import { action, internalMutation, query, mutation } from "./_generated/server";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";
import { internal } from "./_generated/api";

/**
 * WebSub (PubSubHubbub) — instant push for feeds that support it, with the
 * conditional-polling cron as the fallback. The hub verifies via GET /websub
 * and pushes updates via POST /websub (see http.ts).
 */
export const subscribe = action({
  args: { sourceId: v.string(), topicUrl: v.string(), hubUrl: v.string() },
  handler: async (ctx, { sourceId, topicUrl, hubUrl }): Promise<{ ok: boolean }> => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const site = process.env.CONVEX_SITE_URL ?? "";
    const callback = `${site}/websub?sourceId=${encodeURIComponent(sourceId)}`;
    const secret = Math.abs(hashStr(sourceId + site)).toString(16);
    await ctx.runMutation(internal.websub.upsert, { sourceId, topicUrl, hubUrl, secret });
    try {
      const body = new URLSearchParams({
        "hub.mode": "subscribe",
        "hub.topic": topicUrl,
        "hub.callback": callback,
        "hub.secret": secret,
        "hub.verify": "async",
      });
      const res = await fetch(hubUrl, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body,
      });
      await ctx.runMutation(internal.websub.setStatus, { sourceId, status: res.ok ? "active" : "failed" });
      return { ok: res.ok };
    } catch {
      await ctx.runMutation(internal.websub.setStatus, { sourceId, status: "failed" });
      return { ok: false };
    }
  },
});

function hashStr(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return h;
}

export const upsert = internalMutation({
  args: { sourceId: v.string(), topicUrl: v.string(), hubUrl: v.string(), secret: v.string() },
  handler: async (ctx, a) => {
    const ex = await ctx.db
      .query("subscriptions")
      .withIndex("by_sourceId", (q) => q.eq("sourceId", a.sourceId))
      .unique();
    if (ex) await ctx.db.patch(ex._id, { topicUrl: a.topicUrl, hubUrl: a.hubUrl, secret: a.secret });
    else await ctx.db.insert("subscriptions", { ...a, status: "pending" });
  },
});

export const setStatus = internalMutation({
  args: { sourceId: v.string(), status: v.union(v.literal("pending"), v.literal("active"), v.literal("failed")) },
  handler: async (ctx, { sourceId, status }) => {
    const ex = await ctx.db
      .query("subscriptions")
      .withIndex("by_sourceId", (q) => q.eq("sourceId", sourceId))
      .unique();
    if (ex) await ctx.db.patch(ex._id, { status, subscribedAt: status === "active" ? Date.now() : ex.subscribedAt });
  },
});

export const recordPing = internalMutation({
  args: { sourceId: v.optional(v.string()) },
  handler: async (ctx, { sourceId }) => {
    if (!sourceId) return;
    const ex = await ctx.db
      .query("subscriptions")
      .withIndex("by_sourceId", (q) => q.eq("sourceId", sourceId))
      .unique();
    if (ex) await ctx.db.patch(ex._id, { lastPingAt: Date.now(), status: "active" });
  },
});

export const list = query({
  args: {},
  handler: async (ctx) => ctx.db.query("subscriptions").collect(),
});

/** Unsubscribe (remove a subscription record). */
export const remove = mutation({
  args: { sourceId: v.string() },
  handler: async (ctx, { sourceId }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const ex = await ctx.db
      .query("subscriptions")
      .withIndex("by_sourceId", (q) => q.eq("sourceId", sourceId))
      .unique();
    if (ex) await ctx.db.delete(ex._id);
  },
});
