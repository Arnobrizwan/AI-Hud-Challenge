import { XMLParser } from "fast-xml-parser";
import type { FetchResult, RawItem, SourceSpec } from "../types";
import { stripHtml } from "../text";

const parser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: "@_",
  textNodeName: "#text",
  trimValues: true,
});

const UA =
  "Mozilla/5.0 (compatible; HUD-NewsBot/1.0; +https://github.com/Arnobrizwan/AI-Hud-Challenge)";

function asText(node: unknown): string {
  if (node == null) return "";
  if (typeof node === "string") return node;
  if (typeof node === "number") return String(node);
  if (typeof node === "object") {
    const o = node as Record<string, unknown>;
    if (typeof o["#text"] === "string") return o["#text"] as string;
    if (typeof o["#text"] === "number") return String(o["#text"]);
  }
  return "";
}

function pickLink(link: unknown): string {
  // RSS: <link>url</link>. Atom: <link href=".." rel="alternate"/> (maybe array).
  if (typeof link === "string") return link;
  if (Array.isArray(link)) {
    const alt = link.find(
      (l) => (l as Record<string, unknown>)?.["@_rel"] === "alternate",
    );
    const chosen = (alt ?? link[0]) as Record<string, unknown>;
    return (chosen?.["@_href"] as string) ?? asText(chosen);
  }
  if (link && typeof link === "object") {
    const o = link as Record<string, unknown>;
    return (o["@_href"] as string) ?? asText(o);
  }
  return "";
}

function pickImage(item: Record<string, unknown>): string | undefined {
  const enc = item["enclosure"] as Record<string, unknown> | undefined;
  if (enc?.["@_url"] && String(enc["@_type"] ?? "").startsWith("image"))
    return enc["@_url"] as string;
  const media = item["media:content"] as Record<string, unknown> | undefined;
  if (media?.["@_url"]) return media["@_url"] as string;
  const thumb = item["media:thumbnail"] as Record<string, unknown> | undefined;
  if (thumb?.["@_url"]) return thumb["@_url"] as string;
  // try first <img> in the content
  const html = asText(item["content:encoded"]) || asText(item["description"]);
  const m = html.match(/<img[^>]+src=["']([^"']+)["']/i);
  return m?.[1];
}

function parseDate(...candidates: unknown[]): number | undefined {
  for (const c of candidates) {
    const s = asText(c);
    if (!s) continue;
    const t = Date.parse(s);
    if (!Number.isNaN(t)) return t;
  }
  return undefined;
}

/** Generic RSS 2.0 / Atom adapter. Used for rss, reddit, x, newsletter kinds. */
export async function fetchRss(source: SourceSpec): Promise<FetchResult> {
  const headers: Record<string, string> = { "User-Agent": UA, Accept: "application/rss+xml, application/atom+xml, application/xml, text/xml, */*" };
  if (source.etag) headers["If-None-Match"] = source.etag;
  if (source.lastModified) headers["If-Modified-Since"] = source.lastModified;

  // Reddit aggressively rate-limits datacenter IPs; retry a couple of times
  // with backoff. Other sources fetch once.
  const attempts = source.kind === "reddit" || source.kind === "x" ? 3 : 1;
  let res: Response | null = null;
  for (let i = 0; i < attempts; i++) {
    try {
      res = await fetch(source.url, { headers, redirect: "follow" });
    } catch (e) {
      if (i === attempts - 1) return { items: [], error: `network: ${(e as Error).message}` };
      res = null;
    }
    if (res && res.status !== 429 && res.status !== 503) break;
    if (i < attempts - 1) await new Promise((r) => setTimeout(r, 1200 * (i + 1)));
  }
  if (!res) return { items: [], error: "no response" };
  if (res.status === 304) return { items: [], notModified: true };
  if (!res.ok) return { items: [], error: `http ${res.status}` };

  const xml = await res.text();
  let doc: Record<string, unknown>;
  try {
    doc = parser.parse(xml) as Record<string, unknown>;
  } catch (e) {
    return { items: [], error: `parse: ${(e as Error).message}` };
  }

  const rss = doc["rss"] as Record<string, unknown> | undefined;
  const feed = doc["feed"] as Record<string, unknown> | undefined;
  let rawEntries: unknown[] = [];
  if (rss) {
    const channel = rss["channel"] as Record<string, unknown> | undefined;
    rawEntries = toArray(channel?.["item"]);
  } else if (feed) {
    rawEntries = toArray(feed["entry"]);
  }

  const items: RawItem[] = [];
  for (const e of rawEntries) {
    const item = e as Record<string, unknown>;
    const title = asText(item["title"]).trim();
    const url = pickLink(item["link"]).trim();
    if (!title || !url) continue;
    const html =
      asText(item["content:encoded"]) ||
      asText(item["content"]) ||
      asText(item["summary"]) ||
      asText(item["description"]);
    items.push({
      sourceId: source.sourceId,
      sourceName: source.name,
      kind: source.kind,
      sourceWeight: source.weight,
      title: stripHtml(title),
      url,
      contentHtml: html || undefined,
      contentText: stripHtml(html),
      author:
        asText(item["dc:creator"]) ||
        asText((item["author"] as Record<string, unknown>)?.["name"]) ||
        asText(item["author"]) ||
        undefined,
      image: pickImage(item),
      publishedAt: parseDate(
        item["pubDate"],
        item["published"],
        item["updated"],
        item["dc:date"],
      ),
      points: 0,
      comments: 0,
      sourceTopics: source.topics,
    });
  }

  return {
    items,
    etag: res.headers.get("etag") ?? undefined,
    lastModified: res.headers.get("last-modified") ?? undefined,
  };
}

function toArray(x: unknown): unknown[] {
  if (x == null) return [];
  return Array.isArray(x) ? x : [x];
}
