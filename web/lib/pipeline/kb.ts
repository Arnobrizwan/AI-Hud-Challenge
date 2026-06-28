/**
 * Stage 3b — knowledge-base linking. Resolves entity surface names to Wikidata
 * QIDs via the free wbsearchentities API. Best-effort + capped, with an
 * in-batch cache so we never hammer the API. Runs inside a Convex action.
 */

const WIKIDATA = "https://www.wikidata.org/w/api.php";

export async function linkEntity(name: string): Promise<string | null> {
  try {
    const url =
      `${WIKIDATA}?action=wbsearchentities&search=${encodeURIComponent(name)}` +
      `&language=en&format=json&limit=1&origin=*`;
    const res = await fetch(url, { headers: { Accept: "application/json" } });
    if (!res.ok) return null;
    const data = (await res.json()) as { search?: { id?: string }[] };
    return data.search?.[0]?.id ?? null;
  } catch {
    return null;
  }
}

/** Link a list of entity names, capped, with a shared cache. */
export async function linkEntities(
  names: string[],
  cache: Map<string, string | null>,
  cap = 6,
): Promise<{ name: string; qid: string }[]> {
  const out: { name: string; qid: string }[] = [];
  let calls = 0;
  for (const name of names.slice(0, cap)) {
    if (out.length >= cap) break;
    let qid = cache.get(name);
    if (qid === undefined) {
      if (calls >= cap) break;
      calls++;
      qid = await linkEntity(name);
      cache.set(name, qid);
    }
    if (qid) out.push({ name, qid });
  }
  return out;
}
