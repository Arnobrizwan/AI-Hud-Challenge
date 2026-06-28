/** Pure text helpers shared by ingest / normalize / enrich / dedup / summarize. */

const HTML_ENTITIES: Record<string, string> = {
  "&amp;": "&",
  "&lt;": "<",
  "&gt;": ">",
  "&quot;": '"',
  "&#39;": "'",
  "&apos;": "'",
  "&nbsp;": " ",
  "&hellip;": "…",
  "&mdash;": "—",
  "&ndash;": "–",
  "&rsquo;": "'",
  "&lsquo;": "'",
  "&ldquo;": '"',
  "&rdquo;": '"',
};

export function decodeEntities(s: string): string {
  return s
    .replace(/&#(\d+);/g, (_, d) => String.fromCharCode(parseInt(d, 10)))
    .replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCharCode(parseInt(h, 16)))
    .replace(/&[a-z]+;/gi, (m) => HTML_ENTITIES[m] ?? m);
}

export function stripHtml(html: string | undefined | null): string {
  if (!html) return "";
  return decodeEntities(
    html
      .replace(/<script[\s\S]*?<\/script>/gi, " ")
      .replace(/<style[\s\S]*?<\/style>/gi, " ")
      .replace(/<[^>]+>/g, " "),
  )
    .replace(/\s+/g, " ")
    .trim();
}

export function wordCount(text: string): number {
  if (!text) return 0;
  return text.split(/\s+/).filter(Boolean).length;
}

/** Lowercased, punctuation-stripped title for dedupe keys + similarity. */
export function normalizeTitle(title: string): string {
  return title
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^\w\s]/g, " ")
    .replace(/\b(the|a|an|to|of|in|on|for|and|is|are|how|why|what)\b/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

const STOP = new Set([
  "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
  "is", "are", "was", "were", "be", "been", "as", "at", "by", "it", "its",
  "this", "that", "these", "those", "from", "how", "why", "what", "we", "you",
  "your", "they", "their", "has", "have", "will", "can", "new",
]);

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOP.has(w));
}

/** First ~N chars at a sentence/word boundary — extractive teaser. */
export function extractiveSummary(text: string, max = 160): string {
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean.length <= max) return clean;
  const slice = clean.slice(0, max);
  const lastStop = Math.max(slice.lastIndexOf(". "), slice.lastIndexOf("! "), slice.lastIndexOf("? "));
  if (lastStop > max * 0.5) return slice.slice(0, lastStop + 1).trim();
  const lastSpace = slice.lastIndexOf(" ");
  return (lastSpace > 0 ? slice.slice(0, lastSpace) : slice).trim() + "…";
}

/** 32-bit hash of a token (FNV-1a). */
function fnv1a(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * 64-bit SimHash (as 16-hex) over tokens — borderline near-dup fallback to
 * complement MinHash/LSH. Similar texts → small Hamming distance.
 */
export function simHash(tokens: string[]): string {
  const bits = new Array(64).fill(0);
  for (const t of tokens) {
    const h1 = fnv1a(t);
    const h2 = fnv1a(t + "#");
    for (let i = 0; i < 32; i++) bits[i] += (h1 >>> i) & 1 ? 1 : -1;
    for (let i = 0; i < 32; i++) bits[32 + i] += (h2 >>> i) & 1 ? 1 : -1;
  }
  // pack into hex
  let hex = "";
  for (let nib = 0; nib < 16; nib++) {
    let v = 0;
    for (let b = 0; b < 4; b++) if (bits[nib * 4 + b] > 0) v |= 1 << b;
    hex += v.toString(16);
  }
  return hex;
}

export function hammingHex(a: string, b: string): number {
  if (a.length !== b.length) return 64;
  let d = 0;
  for (let i = 0; i < a.length; i++) {
    let x = parseInt(a[i], 16) ^ parseInt(b[i], 16);
    while (x) { d += x & 1; x >>= 1; }
  }
  return d;
}

/** Fixed-dim hashing-trick vector (L2-normalized) — a lightweight "embedding". */
export function hashingVector(tokens: string[], dim = 64): number[] {
  const v = new Array(dim).fill(0);
  for (const t of tokens) {
    const h = fnv1a(t);
    const idx = h % dim;
    v[idx] += (h & 1) ? 1 : -1;
  }
  let norm = 0;
  for (const x of v) norm += x * x;
  norm = Math.sqrt(norm) || 1;
  return v.map((x) => x / norm);
}

export function cosine(a: number[], b: number[]): number {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

/**
 * Readability-style main-content extraction: drop nav/boilerplate blocks and
 * keep the longest text-dense paragraphs. Heuristic fallback (no DOM).
 */
export function readabilityExtract(html: string | undefined | null): string {
  if (!html) return "";
  const blocks = html
    .replace(/<(nav|header|footer|aside|script|style|form)[\s\S]*?<\/\1>/gi, " ")
    .split(/<\/?(?:p|div|section|article|br|li)[^>]*>/i)
    .map((b) => stripHtml(b))
    .filter((b) => b.split(/\s+/).length >= 8); // text-dense only
  const text = blocks.join("\n\n").replace(/\n{3,}/g, "\n\n").trim();
  return text;
}

/** Stable 53-bit hash → hex string (djb2-xor variant). No crypto needed. */
export function hashString(s: string): string {
  let h1 = 0xdeadbeef ^ s.length;
  let h2 = 0x41c6ce57 ^ s.length;
  for (let i = 0; i < s.length; i++) {
    const ch = s.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  const out = 4294967296 * (2097151 & h2) + (h1 >>> 0);
  return out.toString(16);
}
