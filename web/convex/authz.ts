import { getAuthUserId } from "@convex-dev/auth/server";
import { query, internalQuery, type QueryCtx, type MutationCtx } from "./_generated/server";
import { Id } from "./_generated/dataModel";

/**
 * Role-based access control for the operator console.
 *
 * A user is an admin if EITHER their (non-guest) email is in the `ADMIN_EMAILS`
 * env allowlist (the bootstrap path — set via `npx convex env set ADMIN_EMAILS`)
 * OR they have a row in the `userRoles` table. Anonymous guests have no email
 * and no row, so they are never admins.
 *
 * Operator mutations/actions call `requireAdmin`; read-only operator data is
 * additionally hidden in the UI for non-admins.
 */

function adminEmails(): string[] {
  return (process.env.ADMIN_EMAILS ?? "")
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);
}

export async function isAdmin(ctx: QueryCtx | MutationCtx): Promise<boolean> {
  const userId = await getAuthUserId(ctx);
  if (!userId) return false;
  const user = await ctx.db.get(userId);
  const email = (user as { email?: string } | null)?.email?.toLowerCase();
  if (email && adminEmails().includes(email)) return true;
  const role = await ctx.db
    .query("userRoles")
    .withIndex("by_user", (q) => q.eq("userId", userId))
    .unique();
  return role?.role === "admin";
}

/** Throw unless the caller is an authenticated admin. Returns their userId. */
export async function requireAdmin(ctx: QueryCtx | MutationCtx): Promise<Id<"users">> {
  const userId = await getAuthUserId(ctx);
  if (!userId) throw new Error("Not authenticated");
  if (!(await isAdmin(ctx))) throw new Error("Admin access required");
  return userId;
}

/** Client-facing: is the current user an admin? (drives nav + console gating) */
export const amIAdmin = query({
  args: {},
  handler: async (ctx) => isAdmin(ctx),
});

/** Internal variant so actions (no db) can authorize via runQuery. */
export const amIAdminInternal = internalQuery({
  args: {},
  handler: async (ctx) => isAdmin(ctx),
});
