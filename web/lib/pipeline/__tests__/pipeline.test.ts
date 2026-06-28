import { describe, it, expect } from "vitest";
import {
  normalizeTitle, extractiveSummary, simHash, hammingHex,
  hashingVector, cosine, readabilityExtract, hashString, tokenize,
} from "../text";
import { canonicalizeUrl, normalizeBatch } from "../normalize";
import { dedupCluster, isBorderlineDuplicate } from "../dedup";
import { recencyScore, scoreForUser, computeItemFeatures, topicalMatch } from "../rank";
import { enrichBatch } from "../enrich";
import { isGrounded, buildExtractive } from "../summarize";
import type { EnrichedItem, RawItem } from "../types";

// ---- text utilities --------------------------------------------------------
describe("text", () => {
  it("normalizeTitle lowercases + strips stopwords/punctuation", () => {
    expect(normalizeTitle("The Future of AI: A Guide!")).toBe("future ai guide");
  });
  it("extractiveSummary bounds length", () => {
    const s = extractiveSummary("word ".repeat(200), 120);
    expect(s.length).toBeLessThanOrEqual(123);
  });
  it("hashString is stable + differs by input", () => {
    expect(hashString("abc")).toBe(hashString("abc"));
    expect(hashString("abc")).not.toBe(hashString("abd"));
  });
  it("simHash: similar texts have small Hamming distance, different large", () => {
    const a = simHash(tokenize("openai releases new gpt model with better reasoning"));
    const b = simHash(tokenize("openai releases a new gpt model with improved reasoning"));
    const c = simHash(tokenize("apple announces a new iphone with a faster chip"));
    expect(hammingHex(a, b)).toBeLessThan(hammingHex(a, c));
  });
  it("hashingVector + cosine: similar > dissimilar", () => {
    const a = hashingVector(tokenize("machine learning model training data"));
    const b = hashingVector(tokenize("machine learning model training dataset"));
    const c = hashingVector(tokenize("cooking recipe pasta tomato sauce"));
    expect(cosine(a, b)).toBeGreaterThan(cosine(a, c));
  });
  it("readabilityExtract drops boilerplate, keeps dense text", () => {
    const html = "<nav>home about</nav><p>This is a sufficiently long paragraph of real content here.</p><footer>copyright</footer>";
    const out = readabilityExtract(html);
    expect(out).toContain("real content");
    expect(out).not.toContain("copyright");
  });
});

// ---- normalize -------------------------------------------------------------
describe("normalize", () => {
  it("canonicalizeUrl strips tracking params + trailing slash + www", () => {
    expect(canonicalizeUrl("https://www.x.com/a/?utm_source=hn&ref=feed&id=5#top"))
      .toBe("https://x.com/a?id=5");
  });
  it("normalizeBatch drops exact duplicates", () => {
    const raw = (url: string, title: string): RawItem => ({
      sourceId: "s", sourceName: "S", kind: "rss", sourceWeight: 0.5,
      title, url, sourceTopics: [],
    });
    const out = normalizeBatch([raw("https://x.com/a", "Hello World"), raw("https://x.com/a", "Hello World")]);
    expect(out).toHaveLength(1);
  });
});

// ---- dedup -----------------------------------------------------------------
function eitem(title: string, sourceWeight = 0.5, publishedAt = Date.now()): EnrichedItem {
  return {
    sourceId: "s", sourceName: "S", kind: "rss", sourceWeight,
    title, url: "https://x.com/" + hashString(title), sourceTopics: [],
    canonicalUrl: "https://x.com/" + hashString(title), dedupeKey: hashString(title),
    summaryExtractive: title, readableText: title, contentHash: hashString(title),
    lang: "en", publishedAt, wordCount: title.split(" ").length,
    topics: ["ai"], entities: [], contentType: "news",
  };
}
describe("dedup", () => {
  it("clusters near-duplicate titles, keeps distinct apart", () => {
    const items = [
      eitem("OpenAI launches GPT-5 with major reasoning gains today"),
      eitem("OpenAI launches GPT-5 with major reasoning improvements today"),
      eitem("Stripe announces new payments API for global merchants"),
    ];
    const { clusters, itemCluster } = dedupCluster(items);
    expect(itemCluster[0]).toBe(itemCluster[1]); // the two GPT-5 items merge
    expect(itemCluster[0]).not.toBe(itemCluster[2]); // Stripe is separate
    expect(clusters.length).toBe(2);
  });
  it("elects representative by source weight", () => {
    const items = [
      eitem("Same big event headline about the thing happening", 0.3),
      eitem("Same big event headline about the thing happening now", 0.9),
    ];
    const { clusters } = dedupCluster(items);
    const rep = clusters[0].representativeIndex;
    expect(items[rep].sourceWeight).toBe(0.9);
  });

  // ---- Stage B borderline confirmation (SimHash + cosine) ----
  it("isBorderlineDuplicate: strong Jaccard groups outright", () => {
    expect(isBorderlineDuplicate(0.6, 40, 0.0)).toBe(true);
  });
  it("isBorderlineDuplicate: clearly-different Jaccard never groups", () => {
    expect(isBorderlineDuplicate(0.1, 0, 1.0)).toBe(false);
  });
  it("isBorderlineDuplicate: borderline needs BOTH SimHash AND cosine to agree", () => {
    expect(isBorderlineDuplicate(0.4, 3, 0.9)).toBe(true); // both pass
    expect(isBorderlineDuplicate(0.4, 30, 0.9)).toBe(false); // SimHash too far
    expect(isBorderlineDuplicate(0.4, 3, 0.4)).toBe(false); // cosine too low
  });
  it("does not over-merge unrelated stories via the borderline path", () => {
    const items = [
      eitem("Apple unveils new M5 chip for the next MacBook Pro lineup"),
      eitem("EU passes sweeping antitrust law targeting big tech platforms"),
    ];
    const { itemCluster } = dedupCluster(items);
    expect(itemCluster[0]).not.toBe(itemCluster[1]);
  });
});

// ---- rank ------------------------------------------------------------------
describe("rank", () => {
  it("recencyScore decays: newer > older", () => {
    const now = Date.now();
    expect(recencyScore(now, 8, now)).toBeGreaterThan(recencyScore(now - 8 * 3600e3, 8, now));
  });
  it("topicalMatch rewards focus overlap", () => {
    expect(topicalMatch(["ai", "llm"], ["ai"], "s", [])).toBeGreaterThan(topicalMatch(["sports"], ["ai"], "s", []));
  });
  it("scoreForUser: focus topic + unseen ranks above off-topic seen; muted suppressed", () => {
    const items = [eitem("a"), eitem("b")];
    const features = computeItemFeatures(items, { halfLifeHours: 8 });
    const weights = { recency: 0.2, sourceWeight: 0.1, topicalMatch: 0.3, novelty: 0.1, velocity: 0.1, popularity: 0.2 };
    const onTopic = scoreForUser({ topics: ["ai"], sourceId: "s", id: "1", features: features[0] },
      { focusTopics: ["ai"], boostedSources: [], mutedSources: [], focusVsPopularMix: 1, seen: new Set() }, weights);
    const offTopic = scoreForUser({ topics: ["sports"], sourceId: "s", id: "2", features: features[1] },
      { focusTopics: ["ai"], boostedSources: [], mutedSources: [], focusVsPopularMix: 1, seen: new Set(["2"]) }, weights);
    expect(onTopic.score).toBeGreaterThan(offTopic.score);
    expect(onTopic.lane).toBe("focus");
    const muted = scoreForUser({ topics: ["ai"], sourceId: "s", id: "1", features: features[0] },
      { focusTopics: ["ai"], boostedSources: [], mutedSources: ["s"], focusVsPopularMix: 1, seen: new Set() }, weights);
    expect(muted.score).toBeLessThan(onTopic.score * 0.01 + 0.001);
  });
  it("popularity normalizes engagement within batch", () => {
    const hot = { ...eitem("hot"), points: 500, comments: 100 };
    const cold = { ...eitem("cold"), points: 0, comments: 0 };
    const f = computeItemFeatures([hot, cold], { halfLifeHours: 8 });
    expect(f[0].popularity).toBeGreaterThan(f[1].popularity);
  });
});

// ---- enrich ----------------------------------------------------------------
describe("enrich", () => {
  it("classifies AI topics + detects discussion content type", () => {
    const n = normalizeBatch([{
      sourceId: "hn", sourceName: "HN", kind: "hackernews", sourceWeight: 0.9,
      title: "New LLM beats GPT-4 on agent benchmarks", url: "https://news.ycombinator.com/x",
      contentText: "A new large language model and ai agent system...", sourceTopics: ["ai"],
    }]);
    const [e] = enrichBatch(n);
    expect(e.topics).toContain("ai");
    expect(e.contentType).toBe("discussion");
  });
});

// ---- summarize -------------------------------------------------------------
describe("summarize", () => {
  it("isGrounded rejects invented entities, accepts grounded", () => {
    const src = "OpenAI released a model in San Francisco.";
    expect(isGrounded("OpenAI released a model in San Francisco", src)).toBe(true);
    expect(isGrounded("Google and Microsoft and Tesla and Amazon launched together", src)).toBe(false);
  });
  it("buildExtractive falls back to title when no body", () => {
    expect(buildExtractive("A headline", "")).toBe("A headline");
  });
});
