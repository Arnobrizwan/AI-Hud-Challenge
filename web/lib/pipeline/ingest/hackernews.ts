import type { FetchResult, RawItem, SourceSpec } from "../types";

interface AlgoliaHit {
  objectID: string;
  title: string | null;
  url: string | null;
  points: number | null;
  num_comments: number | null;
  created_at_i: number | null;
  author: string | null;
  story_text?: string | null;
}

/**
 * HackerNews via the Algolia API — gives real points + comment counts, which
 * the ranker uses as the popularity signal. `front_page` ≈ the live HN home.
 */
export async function fetchHackerNews(source: SourceSpec): Promise<FetchResult> {
  const url = "https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage=50";
  let res: Response;
  try {
    res = await fetch(url, { headers: { Accept: "application/json" } });
  } catch (e) {
    return { items: [], error: `network: ${(e as Error).message}` };
  }
  if (!res.ok) return { items: [], error: `http ${res.status}` };

  const data = (await res.json()) as { hits?: AlgoliaHit[] };
  const items: RawItem[] = [];
  for (const h of data.hits ?? []) {
    if (!h.title) continue;
    const link = h.url ?? `https://news.ycombinator.com/item?id=${h.objectID}`;
    items.push({
      sourceId: source.sourceId,
      sourceName: source.name,
      kind: "hackernews",
      sourceWeight: source.weight,
      title: h.title,
      url: link,
      contentText: h.story_text ?? undefined,
      author: h.author ?? undefined,
      publishedAt: h.created_at_i ? h.created_at_i * 1000 : undefined,
      points: h.points ?? 0,
      comments: h.num_comments ?? 0,
      sourceTopics: source.topics,
    });
  }
  return { items };
}
