import type { NextRequest } from "next/server";
import { api } from "@/convex/_generated/api";
import type { Id } from "@/convex/_generated/dataModel";
import { convexClient } from "@/lib/convexHttp";
import { JSON_HEADERS } from "@/lib/api";

export const dynamic = "force-dynamic";

/**
 * GET /api/cluster?id=  → representative + related members.
 * Proxies the `items.publicCluster` query.
 */
export async function GET(req: NextRequest) {
  const id = req.nextUrl.searchParams.get("id");
  if (!id) {
    return Response.json({ error: "missing id" }, { status: 400, headers: JSON_HEADERS });
  }
  let data: Awaited<ReturnType<typeof fetchCluster>>;
  try {
    data = await fetchCluster(id as Id<"clusters">);
  } catch (e) {
    // Malformed ids reach Convex's validator and throw — surface as 400.
    return Response.json({ error: (e as Error).message }, { status: 400, headers: JSON_HEADERS });
  }
  if (!data) {
    return Response.json({ error: "not found" }, { status: 404, headers: JSON_HEADERS });
  }
  return new Response(JSON.stringify(data), {
    headers: { ...JSON_HEADERS, "Cache-Control": "public, max-age=30" },
  });
}

function fetchCluster(clusterId: Id<"clusters">) {
  return convexClient().query(api.items.publicCluster, { clusterId });
}

export async function OPTIONS() {
  return new Response(null, { headers: JSON_HEADERS });
}
