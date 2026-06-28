import type { GoldEntry } from "../lib/pipeline/evalMetrics";

/**
 * Curated, stratified gold evaluation set. Each entry marks a relevant story by
 * a title substring, graded 0..1. Stratified across the taxonomy so Precision@K
 * / nDCG aren't dominated by a single hot topic. Seeded into the `goldSet`
 * table by `sources.seed` (idempotent) and consumed by `evaluation.runEval`.
 */
export const SEED_GOLD: GoldEntry[] = [
  // ai
  { topic: "ai", keyword: "openai", relevance: 1.0 },
  { topic: "ai", keyword: "gpt", relevance: 1.0 },
  { topic: "ai", keyword: "anthropic", relevance: 1.0 },
  // llm
  { topic: "llm", keyword: "claude", relevance: 1.0 },
  { topic: "llm", keyword: "llama", relevance: 0.9 },
  { topic: "llm", keyword: "language model", relevance: 0.8 },
  // agents
  { topic: "agents", keyword: "agent", relevance: 0.8 },
  { topic: "agents", keyword: "mcp", relevance: 0.9 },
  // security
  { topic: "security", keyword: "vulnerability", relevance: 1.0 },
  { topic: "security", keyword: "cve", relevance: 0.9 },
  { topic: "security", keyword: "breach", relevance: 0.8 },
  // startups / business
  { topic: "startups", keyword: "funding", relevance: 0.8 },
  { topic: "startups", keyword: "series a", relevance: 0.9 },
  { topic: "business", keyword: "acquisition", relevance: 0.7 },
  // hardware
  { topic: "hardware", keyword: "nvidia", relevance: 0.9 },
  { topic: "hardware", keyword: "gpu", relevance: 0.8 },
  // science
  { topic: "science", keyword: "research", relevance: 0.7 },
  { topic: "science", keyword: "quantum", relevance: 0.8 },
  // programming / open-source
  { topic: "programming", keyword: "typescript", relevance: 0.8 },
  { topic: "open-source", keyword: "open source", relevance: 0.7 },
];
