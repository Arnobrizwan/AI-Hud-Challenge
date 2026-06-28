/**
 * Pure helpers shared by the Next.js `/api/*` route handlers (app/api/*). Kept
 * framework-free so they're unit-testable without a running Convex/Next server.
 */

/** Hard cap so a caller can't request an unbounded page. */
export const FEED_MAX_LIMIT = 100;

/** CORS + JSON headers mirroring the Convex `.site` REST contract (convex/http.ts). */
export const JSON_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, If-None-Match",
  "Content-Type": "application/json",
};

/** Parse `?n=` into a bounded integer, falling back to `dflt` on junk input. */
export function clampInt(raw: string | null, dflt: number, min: number, max: number): number {
  const n = raw == null ? NaN : Number(raw);
  if (!Number.isFinite(n)) return dflt;
  return Math.max(min, Math.min(max, Math.trunc(n)));
}

export interface FeedParams {
  topic: string | undefined;
  limit: number;
  offset: number;
}

/** Read `topic` / `limit` / `offset` from a feed request's query string. */
export function parseFeedParams(sp: URLSearchParams): FeedParams {
  const topic = sp.get("topic") || undefined;
  return {
    topic,
    limit: clampInt(sp.get("limit"), 30, 1, FEED_MAX_LIMIT),
    offset: clampInt(sp.get("offset"), 0, 0, 100_000),
  };
}

/** True when the client's `If-None-Match` matches the payload ETag (→ 304). */
export function notModified(reqEtag: string | null, etag: string | undefined): boolean {
  return !!reqEtag && !!etag && reqEtag === etag;
}

/** Strip a `Bearer ` prefix from an Authorization header → bare token (or undefined). */
export function bearerToken(authHeader: string | null): string | undefined {
  if (!authHeader) return undefined;
  const m = /^Bearer\s+(.+)$/i.exec(authHeader.trim());
  return m ? m[1] : authHeader.trim() || undefined;
}
