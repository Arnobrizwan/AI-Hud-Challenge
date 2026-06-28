/**
 * Next.js 16 "Proxy" (formerly Middleware). Convex Auth handles session
 * refresh here and gates the authenticated areas of the HUD.
 */
import {
  convexAuthNextjsMiddleware,
  createRouteMatcher,
  nextjsMiddlewareRedirect,
} from "@convex-dev/auth/nextjs/server";

const isSignInPage = createRouteMatcher(["/signin"]);
const isProtected = createRouteMatcher([
  "/feed(.*)",
  "/bookmarks(.*)",
  "/settings(.*)",
  "/dashboard(.*)",
]);

export default convexAuthNextjsMiddleware(async (request, { convexAuth }) => {
  const authed = await convexAuth.isAuthenticated();
  if (isSignInPage(request) && authed) {
    return nextjsMiddlewareRedirect(request, "/feed");
  }
  if (isProtected(request) && !authed) {
    return nextjsMiddlewareRedirect(request, "/signin");
  }
});

export const config = {
  // Run on everything except static files and Next internals.
  matcher: ["/((?!.*\\..*|_next).*)", "/", "/(api|trpc)(.*)"],
};
