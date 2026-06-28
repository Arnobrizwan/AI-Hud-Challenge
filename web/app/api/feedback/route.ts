import type { NextRequest } from "next/server";
import { api } from "@/convex/_generated/api";
import type { Id } from "@/convex/_generated/dataModel";
import { convexClient } from "@/lib/convexHttp";
import { JSON_HEADERS, bearerToken } from "@/lib/api";

export const dynamic = "force-dynamic";

const ACTIONS = new Set([
  "up", "down", "not_interested", "mute_source", "click", "dwell", "seen", "more_like_this",
]);
type FeedbackAction =
  | "up" | "down" | "not_interested" | "mute_source" | "click" | "dwell" | "seen" | "more_like_this";

/**
 * POST /api/feedback {itemId, action, value?}
 * Proxies the authenticated `feedback.record` mutation. The caller's Convex
 * auth token must be forwarded as `Authorization: Bearer <token>`.
 */
export async function POST(req: NextRequest) {
  const body = (await req.json().catch(() => null)) as
    | { itemId?: string; action?: string; value?: number }
    | null;
  if (!body?.itemId || !body?.action) {
    return Response.json({ error: "itemId+action required" }, { status: 400, headers: JSON_HEADERS });
  }
  if (!ACTIONS.has(body.action)) {
    return Response.json({ error: `invalid action: ${body.action}` }, { status: 400, headers: JSON_HEADERS });
  }
  const token = bearerToken(req.headers.get("authorization"));
  if (!token) {
    return Response.json({ error: "authentication required" }, { status: 401, headers: JSON_HEADERS });
  }
  try {
    await convexClient(token).mutation(api.feedback.record, {
      itemId: body.itemId as Id<"items">,
      action: body.action as FeedbackAction,
      value: body.value,
    });
    return Response.json({ ok: true }, { headers: JSON_HEADERS });
  } catch (e) {
    return Response.json({ error: (e as Error).message }, { status: 401, headers: JSON_HEADERS });
  }
}

export async function OPTIONS() {
  return new Response(null, { headers: JSON_HEADERS });
}
