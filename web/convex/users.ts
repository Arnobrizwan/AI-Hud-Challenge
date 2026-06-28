import { query } from "./_generated/server";
import { getAuthUserId } from "@convex-dev/auth/server";
import { isAdmin } from "./authz";

/** The signed-in user (or null). */
export const currentUser = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return null;
    const user = await ctx.db.get(userId);
    if (!user) return null;
    return {
      _id: user._id,
      name: user.name ?? null,
      email: user.email ?? null,
      isAnonymous: (user as { isAnonymous?: boolean }).isAnonymous ?? false,
      isAdmin: await isAdmin(ctx),
    };
  },
});
