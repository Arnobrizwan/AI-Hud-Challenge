import { query, mutation, action, internalMutation, internalQuery } from "./_generated/server";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";
import { internal } from "./_generated/api";
import { encryptSecret } from "./crypto";

const providerValidator = v.union(v.literal("openai"), v.literal("anthropic"));

/** Masked list of the user's stored keys (never returns plaintext/ciphertext). */
export const listKeys = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return [];
    const rows = await ctx.db
      .query("apiKeys")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .collect();
    return rows.map((r) => ({
      provider: r.provider,
      last4: r.last4,
      model: r.model ?? null,
      valid: r.valid,
      updatedAt: r.updatedAt,
    }));
  },
});

/** Validate a provider key by calling its list-models endpoint. */
async function validateKey(
  provider: "openai" | "anthropic",
  key: string,
): Promise<boolean> {
  try {
    if (provider === "openai") {
      const res = await fetch("https://api.openai.com/v1/models", {
        headers: { Authorization: `Bearer ${key}` },
      });
      return res.ok;
    }
    const res = await fetch("https://api.anthropic.com/v1/models", {
      headers: { "x-api-key": key, "anthropic-version": "2023-06-01" },
    });
    return res.ok;
  } catch {
    return false;
  }
}

/** Encrypt + validate + persist a BYO key. Returns whether it validated. */
export const saveKey = action({
  args: { provider: providerValidator, key: v.string(), model: v.optional(v.string()) },
  handler: async (ctx, { provider, key, model }): Promise<{ valid: boolean }> => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const trimmed = key.trim();
    if (trimmed.length < 10) throw new Error("Key looks too short");

    const valid = await validateKey(provider, trimmed);
    const ciphertext = await encryptSecret(trimmed);
    const last4 = trimmed.slice(-4);
    await ctx.runMutation(internal.apiKeys.store, {
      userId,
      provider,
      ciphertext,
      last4,
      model,
      valid,
    });
    return { valid };
  },
});

/** Test a key WITHOUT storing it (the "Test key" button). */
export const testKey = action({
  args: { provider: providerValidator, key: v.string() },
  handler: async (ctx, { provider, key }): Promise<{ valid: boolean }> => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    return { valid: await validateKey(provider, key.trim()) };
  },
});

export const deleteKey = mutation({
  args: { provider: providerValidator },
  handler: async (ctx, { provider }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const row = await ctx.db
      .query("apiKeys")
      .withIndex("by_user_provider", (q) =>
        q.eq("userId", userId).eq("provider", provider),
      )
      .unique();
    if (row) await ctx.db.delete(row._id);
  },
});

// ---- internal --------------------------------------------------------------

export const store = internalMutation({
  args: {
    userId: v.id("users"),
    provider: providerValidator,
    ciphertext: v.string(),
    last4: v.string(),
    model: v.optional(v.string()),
    valid: v.boolean(),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("apiKeys")
      .withIndex("by_user_provider", (q) =>
        q.eq("userId", args.userId).eq("provider", args.provider),
      )
      .unique();
    const doc = { ...args, updatedAt: Date.now() };
    if (existing) await ctx.db.patch(existing._id, doc);
    else await ctx.db.insert("apiKeys", doc);
  },
});

/** Internal: ciphertext for a user+provider (decrypted by the calling action). */
export const getCipher = internalQuery({
  args: { userId: v.id("users"), provider: providerValidator },
  handler: async (ctx, { userId, provider }) => {
    const row = await ctx.db
      .query("apiKeys")
      .withIndex("by_user_provider", (q) =>
        q.eq("userId", userId).eq("provider", provider),
      )
      .unique();
    if (!row || !row.valid) return null;
    return { ciphertext: row.ciphertext, model: row.model ?? null };
  },
});
