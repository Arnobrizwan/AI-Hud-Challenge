import { mutation } from "./_generated/server";
import { getAuthUserId } from "@convex-dev/auth/server";

/**
 * GDPR/CCPA right-to-erasure: delete ALL of the signed-in user's data, then the
 * account itself. Irreversible. Returns counts of what was removed.
 */
export const deleteMyAccount = mutation({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    let removed = 0;

    const delByUserIndex = async (
      table: "userPrefs" | "apiKeys" | "bookmarks" | "feedback" | "notificationsLog",
    ) => {
      const rows = await ctx.db
        .query(table)
        .withIndex("by_user", (q) => q.eq("userId", userId))
        .collect();
      for (const r of rows) {
        await ctx.db.delete(r._id);
        removed++;
      }
    };
    await delByUserIndex("userPrefs");
    await delByUserIndex("apiKeys");
    await delByUserIndex("bookmarks");
    await delByUserIndex("feedback");
    await delByUserIndex("notificationsLog");

    // scores (by_user_score), labels (scan), and auth records
    for (const s of await ctx.db.query("scores").withIndex("by_user_score", (q) => q.eq("userId", userId)).collect()) {
      await ctx.db.delete(s._id); removed++;
    }
    for (const l of await ctx.db.query("labels").collect()) {
      if (l.userId === userId) { await ctx.db.delete(l._id); removed++; }
    }
    // Convex Auth records
    for (const a of await ctx.db.query("authAccounts").withIndex("userIdAndProvider", (q) => q.eq("userId", userId)).collect()) {
      await ctx.db.delete(a._id); removed++;
    }
    for (const s of await ctx.db.query("authSessions").withIndex("userId", (q) => q.eq("userId", userId)).collect()) {
      await ctx.db.delete(s._id); removed++;
    }
    await ctx.db.delete(userId);
    removed++;
    return { removed };
  },
});
