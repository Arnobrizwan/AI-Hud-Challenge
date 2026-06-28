import { getAuthUserId } from "@convex-dev/auth/server";
import {
  query, mutation, internalQuery, internalMutation,
  type QueryCtx, type MutationCtx,
} from "./_generated/server";
import { v } from "convex/values";
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

// ---- Granting / revoking admin (userRoles table) --------------------------

async function userByEmail(ctx: QueryCtx | MutationCtx, email: string) {
  const lower = email.trim().toLowerCase();
  if (!lower) return null;
  const users = await ctx.db.query("users").collect();
  return users.find((u) => ((u as { email?: string }).email ?? "").toLowerCase() === lower) ?? null;
}

async function grantRole(ctx: MutationCtx, userId: Id<"users">): Promise<boolean> {
  const existing = await ctx.db
    .query("userRoles")
    .withIndex("by_user", (q) => q.eq("userId", userId))
    .unique();
  if (existing) return false; // already an admin
  await ctx.db.insert("userRoles", { userId, role: "admin", grantedAt: Date.now() });
  return true;
}

/** Admin-only: promote another user to admin by email (they must have signed in once). */
export const grantAdmin = mutation({
  args: { email: v.string() },
  handler: async (ctx, { email }) => {
    await requireAdmin(ctx);
    const u = await userByEmail(ctx, email);
    if (!u) throw new Error(`No account with email ${email} — they must sign in once first.`);
    return { userId: u._id, granted: await grantRole(ctx, u._id) };
  },
});

/** Admin-only: revoke an admin's userRoles grant (env ADMIN_EMAILS still wins). */
export const revokeAdmin = mutation({
  args: { email: v.string() },
  handler: async (ctx, { email }) => {
    await requireAdmin(ctx);
    const u = await userByEmail(ctx, email);
    if (!u) throw new Error(`No account with email ${email}.`);
    const row = await ctx.db
      .query("userRoles")
      .withIndex("by_user", (q) => q.eq("userId", u._id))
      .unique();
    if (row) await ctx.db.delete(row._id);
    return { userId: u._id, revoked: !!row };
  },
});

/** Admin-only: list granted admins (does not include env-allowlisted accounts). */
export const listAdmins = query({
  args: {},
  handler: async (ctx) => {
    await requireAdmin(ctx);
    const roles = await ctx.db.query("userRoles").collect();
    const out: { userId: Id<"users">; email: string | null; grantedAt: number }[] = [];
    for (const r of roles) {
      const u = await ctx.db.get(r.userId);
      out.push({ userId: r.userId, email: (u as { email?: string } | null)?.email ?? null, grantedAt: r.grantedAt });
    }
    return out;
  },
});

/**
 * Bootstrap path (CLI only — no caller identity required, so it works with zero
 * existing admins): npx convex run authz:grantAdminByEmail '{"email":"..."}' [--prod]
 */
export const grantAdminByEmail = internalMutation({
  args: { email: v.string() },
  handler: async (ctx, { email }) => {
    const u = await userByEmail(ctx, email);
    if (!u) throw new Error(`No account with email ${email} — they must sign in once first.`);
    return { userId: u._id, granted: await grantRole(ctx, u._id) };
  },
});
