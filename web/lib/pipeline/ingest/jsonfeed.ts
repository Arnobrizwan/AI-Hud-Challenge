import type { FetchResult, RawItem, SourceSpec } from "../types";
import { stripHtml } from "../text";

/**
 * JSON Feed adapter (https://jsonfeed.org/version/1.1). Parses the top-level
 * `items[]` array into RawItem[]. Mirrors rss.ts: real User-Agent + conditional
 * GET (ETag / If-Modified-Since), 304 short-circuit, per-source health passthrough.
 */

const UA =
  "Mozilla/5.0 (compatible; HUD-NewsBot/1.0; +https://github.com/Arnobrizwan/AI-Hud-Challenge)";

interface JsonFeedAuthor {
  name?: string;
  url?: string;
}
interface JsonFeedItem {
  id?: string | number;
  url?: string;
  external_url?: string;
  title?: string;
  content_html?: string;
  content_text?: string;
  summary?: string;
  image?: string;
  banner_image?: string;
  date_published?: string;
  date_modified?: string;
  author?: JsonFeedAuthor;
  authors?: JsonFeedAuthor[];
  tags?: string[];
}
interface JsonFeedDoc {
  version?: string;
  title?: string;
  items?: JsonFeedItem[];
}

function looksLikeUrl(s: string): boolean {
  return /^https?:\/\//i.test(s);
}

function firstWords(text: string, n: number): string {
  const words = text.split(/\s+/).filter(Boolean).slice(0, n);
  return words.join(" ");
}

function parseDate(...candidates: (string | undefined)[]): number | undefined {
  for (const c of candidates) {
    if (!c) continue;
    const t = Date.parse(c);
    if (!Number.isNaN(t)) return t;
  }
  return undefined;
}

/** Pure parse step (no network) — turns a parsed JSON Feed doc into RawItem[]. */
export function parseJsonFeed(doc: JsonFeedDoc, source: SourceSpec): RawItem[] {
  const entries = Array.isArray(doc?.items) ? doc.items : [];
  const items: RawItem[] = [];
  for (const it of entries) {
    const id = it.id != null ? String(it.id) : "";
    const url = (it.url || it.external_url || (looksLikeUrl(id) ? id : "")).trim();
    if (!url) continue;
    const html = it.content_html || "";
    const text = it.content_text || it.summary || stripHtml(html);
    // JSON Feed allows title-less microblog items — synthesize from content.
    const title = (it.title?.trim() || firstWords(stripHtml(text), 12)).trim();
    if (!title) continue;
    const author = it.authors?.[0]?.name || it.author?.name || undefined;
    items.push({
      sourceId: source.sourceId,
      sourceName: source.name,
      kind: source.kind,
      sourceWeight: source.weight,
      title: stripHtml(title),
      url,
      contentHtml: html || undefined,
      contentText: stripHtml(text),
      author,
      image: it.image || it.banner_image || undefined,
      publishedAt: parseDate(it.date_published, it.date_modified),
      points: 0,
      comments: 0,
      sourceTopics: source.topics,
    });
  }
  return items;
}

/** JSON Feed ingest adapter (kind === "jsonfeed"). */
export async function fetchJsonFeed(source: SourceSpec): Promise<FetchResult> {
  const headers: Record<string, string> = {
    "User-Agent": UA,
    Accept: "application/feed+json, application/json, */*",
  };
  if (source.etag) headers["If-None-Match"] = source.etag;
  if (source.lastModified) headers["If-Modified-Since"] = source.lastModified;

  let res: Response;
  try {
    res = await fetch(source.url, { headers, redirect: "follow" });
  } catch (e) {
    return { items: [], error: `network: ${(e as Error).message}` };
  }
  if (res.status === 304) return { items: [], notModified: true };
  if (!res.ok) return { items: [], error: `http ${res.status}` };

  let doc: JsonFeedDoc;
  try {
    doc = (await res.json()) as JsonFeedDoc;
  } catch (e) {
    return { items: [], error: `parse: ${(e as Error).message}` };
  }

  return {
    items: parseJsonFeed(doc, source),
    etag: res.headers.get("etag") ?? undefined,
    lastModified: res.headers.get("last-modified") ?? undefined,
  };
}
