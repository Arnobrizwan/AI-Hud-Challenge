import type { NextRequest } from "next/server";
import { api } from "@/convex/_generated/api";
import { convexClient } from "@/lib/convexHttp";
import { JSON_HEADERS, notModified, parseFeedParams } from "@/lib/api";

// Reads request-time data (query string / headers) → always dynamic.
export const dynamic = "force-dynamic";

/**
 * GET /api/feed?topic=&limit=&offset=
 * Vercel-domain mirror of the Convex `.site` REST contract: proxies the
 * `feed.publicFeed` query with ETag revalidation + pagination.
 */
export async function GET(req: NextRequest) {
  const { topic, limit, offset } = parseFeedParams(req.nextUrl.searchParams);
  let data: Awaited<ReturnType<typeof fetchFeed>>;
  try {
    data = await fetchFeed(topic, limit, offset);
  } catch (e) {
    return Response.json({ error: (e as Error).message }, { status: 502, headers: JSON_HEADERS });
  }

  if (notModified(req.headers.get("if-none-match"), data.etag)) {
    return new Response(null, { status: 304, headers: { ...JSON_HEADERS, ETag: data.etag } });
  }
  return new Response(JSON.stringify({ ...data, limit, offset }), {
    headers: { ...JSON_HEADERS, ETag: data.etag, "Cache-Control": "public, max-age=30" },
  });
}

function fetchFeed(topic: string | undefined, limit: number, offset: number) {
  return convexClient().query(api.feed.publicFeed, { topic, limit, offset });
}

export async function OPTIONS() {
  return new Response(null, { headers: JSON_HEADERS });
}
