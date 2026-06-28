import { ConvexHttpClient } from "convex/browser";

/**
 * Server-side Convex client for the Next.js `/api/*` route handlers. Talks to
 * the same deployment the browser app uses (NEXT_PUBLIC_CONVEX_URL), so the
 * REST routes proxy the exact queries/mutations the UI reads.
 */
export function convexClient(authToken?: string): ConvexHttpClient {
  const url = process.env.NEXT_PUBLIC_CONVEX_URL;
  if (!url) throw new Error("NEXT_PUBLIC_CONVEX_URL is not set");
  const client = new ConvexHttpClient(url);
  if (authToken) client.setAuth(authToken);
  return client;
}
