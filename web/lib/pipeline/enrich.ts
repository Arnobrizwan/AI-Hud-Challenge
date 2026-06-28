import type { EnrichedItem, NormalizedItem } from "./types";
import { tokenize } from "./text";

/**
 * Stage 3 — enrichment. Rules+keywords baseline for topics, a lightweight
 * capitalized-phrase heuristic for entities, and content-type detection.
 * (Swappable later for an LLM classifier — the stage signature stays the same.)
 */

const TOPIC_KEYWORDS: Record<string, string[]> = {
  ai: ["ai", "artificial intelligence", "openai", "anthropic", "gpt", "claude", "gemini", "deepmind", "chatbot"],
  llm: ["llm", "language model", "gpt", "claude", "llama", "mistral", "prompt", "rag", "fine-tune", "fine tuning", "tokens"],
  ml: ["machine learning", "neural", "training", "dataset", "model", "inference", "pytorch", "tensorflow", "diffusion"],
  agents: ["agent", "agentic", "autonomous", "tool use", "mcp", "workflow", "orchestration"],
  startups: ["startup", "founder", "seed round", "series a", "yc", "y combinator", "vc", "raise", "funding"],
  programming: ["javascript", "typescript", "python", "rust", "golang", "compiler", "framework", "api", "code", "git"],
  "open-source": ["open source", "open-source", "github", "mit license", "apache", "foss", "repo"],
  security: ["security", "vulnerability", "cve", "exploit", "breach", "ransomware", "zero-day", "malware", "phishing"],
  crypto: ["crypto", "bitcoin", "ethereum", "blockchain", "web3", "defi", "token", "nft"],
  science: ["research", "study", "physics", "biology", "quantum", "space", "nasa", "climate", "paper"],
  hardware: ["chip", "gpu", "nvidia", "amd", "tsmc", "silicon", "processor", "hardware", "device", "robot"],
  robotics: ["robot", "robotics", "humanoid", "autonomous vehicle", "drone", "boston dynamics"],
  data: ["database", "data", "analytics", "sql", "warehouse", "etl", "pipeline", "vector"],
  business: ["revenue", "ipo", "acquisition", "earnings", "market", "ceo", "layoffs", "profit"],
  design: ["design", "ux", "ui", "figma", "typography", "interface", "product design"],
  policy: ["regulation", "policy", "law", "eu", "ftc", "antitrust", "ban", "lawsuit", "congress"],
};

const KNOWN_ORGS = [
  "OpenAI", "Anthropic", "Google", "DeepMind", "Meta", "Microsoft", "Apple",
  "Amazon", "Nvidia", "Tesla", "Hugging Face", "Mistral", "Cohere", "Stability AI",
  "GitHub", "Vercel", "Cloudflare", "AWS", "Reddit", "Twitter", "X", "Perplexity",
];

/**
 * Cross-topic aggregators span the whole taxonomy, so their declared topics are
 * NOT a meaningful per-item signal — flooring with them would stamp the source's
 * entire topic set onto every story (making topic tabs show identical results).
 * Topical sources (newsletters, subreddits, blogs) keep the floor.
 */
const AGGREGATOR_KINDS = new Set(["hackernews"]);

function classifyTopics(text: string, sourceTopics: string[], useFloor: boolean): string[] {
  const hay = " " + text.toLowerCase() + " ";
  const scored: [string, number][] = [];
  for (const [topic, kws] of Object.entries(TOPIC_KEYWORDS)) {
    let hits = 0;
    for (const kw of kws) if (hay.includes(" " + kw) || hay.includes(kw + " ")) hits++;
    if (hits > 0) scored.push([topic, hits]);
  }
  scored.sort((a, b) => b[1] - a[1]);
  const top = scored.slice(0, 4).map(([t]) => t);
  // Content-derived topics are authoritative; source topics only floor for
  // topical sources (never for cross-topic aggregators like HackerNews).
  const floor = useFloor ? sourceTopics : [];
  const merged = top.length > 0 ? [...top, ...floor] : floor;
  return Array.from(new Set(merged)).slice(0, 5);
}

/** Recompute topics for an item from its text + source (used at enrich + backfill). */
export function computeTopics(
  title: string,
  text: string,
  sourceTopics: string[],
  kind: string,
): string[] {
  return classifyTopics(title + " " + text, sourceTopics, !AGGREGATOR_KINDS.has(kind));
}

function extractEntities(title: string, text: string): string[] {
  const found = new Set<string>();
  const hay = title + " " + text.slice(0, 600);
  for (const org of KNOWN_ORGS) {
    if (new RegExp(`\\b${org.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(hay)) {
      found.add(org);
    }
  }
  // capitalized bigrams/words in the title (proper nouns)
  const caps = title.match(/\b([A-Z][a-zA-Z0-9.]+(?:\s[A-Z][a-zA-Z0-9.]+){0,2})\b/g) ?? [];
  for (const c of caps) {
    if (c.length > 2 && !["The", "A", "An", "This", "How", "Why"].includes(c)) found.add(c);
    if (found.size >= 8) break;
  }
  return Array.from(found).slice(0, 8);
}

function detectContentType(item: NormalizedItem): EnrichedItem["contentType"] {
  if (item.kind === "reddit" || item.kind === "hackernews" || item.kind === "x")
    return "discussion";
  const t = item.title.toLowerCase();
  if (/(introducing|announc|launch|releas|now available|unveil|ships?)/.test(t))
    return "release";
  if (/(opinion|why |i think|the case for|we should|rant|unpopular)/.test(t))
    return "opinion";
  return "news";
}

/** Stage 3 — enrich one normalized item. */
export function enrichItem(item: NormalizedItem): EnrichedItem {
  const text = item.contentText || item.summaryExtractive || "";
  return {
    ...item,
    topics: computeTopics(item.title, text, item.sourceTopics, item.kind),
    entities: extractEntities(item.title, text),
    contentType: detectContentType(item),
  };
}

export function enrichBatch(items: NormalizedItem[]): EnrichedItem[] {
  return items.map(enrichItem);
}

export { tokenize };
