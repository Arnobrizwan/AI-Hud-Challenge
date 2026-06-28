import { httpRouter } from "convex/server";
import { httpAction } from "./_generated/server";
import { api, internal } from "./_generated/api";
import { auth } from "./auth";
import { Id } from "./_generated/dataModel";

const http = httpRouter();

// Convex Auth routes (sign-in, OAuth callbacks, token refresh).
auth.addHttpRoutes(http);

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, If-None-Match",
  "Content-Type": "application/json",
};

// ---- Real-time interface: documented REST contract (section 14) ------------

// GET /api/feed?topic=&limit=&offset=  → ranked representative items, with ETag.
http.route({
  path: "/api/feed",
  method: "GET",
  handler: httpAction(async (ctx, req) => {
    const u = new URL(req.url);
    const topic = u.searchParams.get("topic") ?? undefined;
    const limit = Number(u.searchParams.get("limit") ?? 30);
    const offset = Number(u.searchParams.get("offset") ?? 0);
    const data = await ctx.runQuery(api.feed.publicFeed, { topic, limit, offset });
    if (req.headers.get("If-None-Match") === data.etag) {
      return new Response(null, { status: 304, headers: { ...CORS, ETag: data.etag } });
    }
    return new Response(JSON.stringify(data), {
      headers: { ...CORS, ETag: data.etag, "Cache-Control": "public, max-age=30" },
    });
  }),
});

// GET /api/cluster?id=  → primary + related members.
http.route({
  path: "/api/cluster",
  method: "GET",
  handler: httpAction(async (ctx, req) => {
    const id = new URL(req.url).searchParams.get("id");
    if (!id) return new Response(JSON.stringify({ error: "missing id" }), { status: 400, headers: CORS });
    const data = await ctx.runQuery(api.items.publicCluster, { clusterId: id as Id<"clusters"> });
    if (!data) return new Response(JSON.stringify({ error: "not found" }), { status: 404, headers: CORS });
    return new Response(JSON.stringify(data), { headers: CORS });
  }),
});

// POST /api/feedback {itemId, action}  → requires Convex auth identity.
http.route({
  path: "/api/feedback",
  method: "POST",
  handler: httpAction(async (ctx, req) => {
    const body = (await req.json().catch(() => null)) as { itemId?: string; action?: string } | null;
    if (!body?.itemId || !body?.action)
      return new Response(JSON.stringify({ error: "itemId+action required" }), { status: 400, headers: CORS });
    try {
      await ctx.runMutation(api.feedback.record, {
        itemId: body.itemId as Id<"items">,
        action: body.action as "up" | "down" | "not_interested" | "click" | "seen",
      });
      return new Response(JSON.stringify({ ok: true }), { headers: CORS });
    } catch (e) {
      return new Response(JSON.stringify({ error: (e as Error).message }), { status: 401, headers: CORS });
    }
  }),
});

http.route({ path: "/api/feed", method: "OPTIONS", handler: httpAction(async () => new Response(null, { headers: CORS })) });
http.route({ path: "/api/feedback", method: "OPTIONS", handler: httpAction(async () => new Response(null, { headers: CORS })) });

// ---- WebSub (section 1): hub verification (GET) + content push (POST) -------
http.route({
  path: "/websub",
  method: "GET",
  handler: httpAction(async (_ctx, req) => {
    const u = new URL(req.url);
    // Echo hub.challenge to confirm the subscription intent.
    const challenge = u.searchParams.get("hub.challenge");
    if (challenge) return new Response(challenge, { status: 200, headers: { "Content-Type": "text/plain" } });
    return new Response("ok", { status: 200 });
  }),
});
http.route({
  path: "/websub",
  method: "POST",
  handler: httpAction(async (ctx, req) => {
    const sourceId = new URL(req.url).searchParams.get("sourceId") ?? undefined;
    // A push notification means the topic changed — kick a pipeline run.
    await ctx.runMutation(internal.websub.recordPing, { sourceId });
    await ctx.runAction(internal.pipeline.runPipeline, { trigger: "websub" });
    return new Response("ok", { status: 202 });
  }),
});

export default http;
