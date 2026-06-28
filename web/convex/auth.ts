import { Password } from "@convex-dev/auth/providers/Password";
import { Anonymous } from "@convex-dev/auth/providers/Anonymous";
import { convexAuth } from "@convex-dev/auth/server";

/**
 * Auth providers:
 *  - Password: email + password accounts (primary).
 *  - Anonymous: one-tap "try the HUD" guest access for reviewers/demo.
 */
export const { auth, signIn, signOut, store, isAuthenticated } = convexAuth({
  providers: [Password, Anonymous],
});
