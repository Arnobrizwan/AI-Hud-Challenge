import type { FetchResult, SourceSpec } from "../types";
import { fetchRss } from "./rss";
import { fetchHackerNews } from "./hackernews";
import { fetchJsonFeed } from "./jsonfeed";

/**
 * Stage 1 — ingest. Dispatches a source to its adapter.
 * Adapters are responsible for conditional GET + returning RawItem[].
 */
export async function ingestSource(source: SourceSpec): Promise<FetchResult> {
  switch (source.kind) {
    case "hackernews":
      return fetchHackerNews(source);
    case "jsonfeed":
      return fetchJsonFeed(source);
    case "rss":
    case "reddit":
    case "x":
    case "newsletter":
      return fetchRss(source);
    default:
      return { items: [], error: `unknown kind: ${source.kind}` };
  }
}

export { fetchRss, fetchHackerNews, fetchJsonFeed };
