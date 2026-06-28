import { describe, it, expect } from "vitest";
import { computeItemFeatures, scoreForUser, type RankWeights, type UserContext } from "../rank";
import { dedupCluster } from "../dedup";
import {
  precisionAtK, ndcgAtK, goldRelevance, pairwiseDupF1, EVAL_GATE_THRESHOLDS,
  evalQualityScore, shouldRollback,
} from "../evalMetrics";
import { hashString } from "../text";
import { SEED_GOLD } from "../../../convex/goldData";
import { DEFAULT_CONFIG, DEFAULT_PREFS } from "../../../convex/defaults";
import type { EnrichedItem } from "../types";

/**
 * ENFORCED CI GATE. Runs the real ranking + dedup over a stratified gold
 * fixture and FAILS if Precision@10 / nDCG@10 / DupF1 fall below their floors.
 * This is the challenge's acceptance criteria turned into a regression guard.
 */

const NOW = 1_750_000_000_000; // fixed clock so the gate is deterministic

function mk(
  title: string,
  topics: string[],
  opts: { weight?: number; points?: number; ageH?: number } = {},
): EnrichedItem {
  const id = hashString(title);
  return {
    sourceId: "src-" + (opts.weight ?? 0.7),
    sourceName: "Src",
    kind: "rss",
    sourceWeight: opts.weight ?? 0.7,
    title,
    url: "https://x.com/" + id,
    canonicalUrl: "https://x.com/" + id,
    dedupeKey: id,
    summaryExtractive: title,
    readableText: title,
    contentHash: id,
    lang: "en",
    publishedAt: NOW - (opts.ageH ?? 2) * 3_600_000,
    wordCount: title.split(" ").length,
    topics,
    entities: [],
    contentType: "news",
    points: opts.points ?? 0,
    comments: 0,
  };
}

// A representative feed: gold-relevant stories (contain gold keywords) mixed
// with off-topic noise the ranker should push down.
const FEED: EnrichedItem[] = [
  mk("OpenAI ships GPT-5 with stronger reasoning", ["ai", "llm"], { weight: 0.9, points: 220 }),
  mk("Anthropic releases new Claude model for agents", ["ai", "llm", "agents"], { weight: 0.9, points: 180 }),
  mk("Critical vulnerability (CVE) found in popular library", ["security"], { weight: 0.8, points: 150 }),
  mk("Nvidia unveils next-gen GPU for AI training", ["hardware", "ai"], { weight: 0.85, points: 140 }),
  mk("New MCP agent framework gains traction", ["agents"], { weight: 0.8, points: 90 }),
  mk("Startup raises Series A funding for LLM tooling", ["startups", "llm"], { weight: 0.75, points: 80 }),
  mk("Researchers publish quantum computing research", ["science"], { weight: 0.8, points: 70 }),
  mk("TypeScript 6 released with faster builds", ["programming"], { weight: 0.7, points: 60 }),
  mk("Open source project hits one million stars", ["open-source"], { weight: 0.7, points: 50 }),
  mk("Local sports team wins championship game", ["business"], { weight: 0.4, points: 5, ageH: 20 }),
  mk("Celebrity gossip column weekly roundup", ["design"], { weight: 0.35, points: 2, ageH: 30 }),
  mk("Weather forecast for the weekend ahead", ["science"], { weight: 0.3, points: 1, ageH: 40 }),
];

describe("eval gate — ranking quality (gold set)", () => {
  const weights: RankWeights = DEFAULT_CONFIG.weights;
  const features = computeItemFeatures(FEED, {
    halfLifeHours: DEFAULT_CONFIG.recencyHalfLifeHours,
    velocityByIndex: FEED.map(() => 1),
    now: NOW,
  });
  const ctx: UserContext = {
    focusTopics: DEFAULT_PREFS.focusTopics,
    boostedSources: [],
    mutedSources: [],
    focusVsPopularMix: 0.6,
    seen: new Set(),
  };
  const ranked = FEED.map((it, i) => ({
    it,
    score: scoreForUser({ topics: it.topics, sourceId: it.sourceId, id: it.url, features: features[i] }, ctx, weights).score,
  })).sort((a, b) => b.score - a.score);

  const rels = ranked.map((r) => goldRelevance(r.it.title, SEED_GOLD));

  it(`Precision@10 >= ${EVAL_GATE_THRESHOLDS.precisionAt10}`, () => {
    const p = precisionAtK(rels, 10);
    expect(p, `Precision@10=${p.toFixed(3)}`).toBeGreaterThanOrEqual(EVAL_GATE_THRESHOLDS.precisionAt10);
  });

  it(`nDCG@10 >= ${EVAL_GATE_THRESHOLDS.ndcgAt10}`, () => {
    const n = ndcgAtK(rels, 10);
    expect(n, `nDCG@10=${n.toFixed(3)}`).toBeGreaterThanOrEqual(EVAL_GATE_THRESHOLDS.ndcgAt10);
  });
});

describe("eval gate — dedup quality (DupF1)", () => {
  // Items with known gold event labels (duplicates share a label).
  const DUP_FEED: Array<{ item: EnrichedItem; event: number }> = [
    { item: mk("OpenAI launches GPT-5 with major reasoning gains today", ["ai"]), event: 1 },
    { item: mk("OpenAI launches GPT-5 with major reasoning improvements today", ["ai"]), event: 1 },
    { item: mk("Nvidia next-gen Blackwell GPU architecture for AI model training now shipping", ["hardware"]), event: 2 },
    { item: mk("Nvidia next-gen Blackwell GPU architecture for AI model training now available", ["hardware"]), event: 2 },
    { item: mk("EU passes sweeping antitrust law targeting big tech platforms", ["policy"]), event: 3 },
    { item: mk("Rust language ships a new borrow checker in latest release", ["programming"]), event: 4 },
  ];
  const { itemCluster } = dedupCluster(DUP_FEED.map((d) => d.item));
  const goldLabels = DUP_FEED.map((d) => d.event);
  const { f1 } = pairwiseDupF1(itemCluster, goldLabels);

  it(`DupF1 >= ${EVAL_GATE_THRESHOLDS.dupF1}`, () => {
    expect(f1, `DupF1=${f1.toFixed(3)}`).toBeGreaterThanOrEqual(EVAL_GATE_THRESHOLDS.dupF1);
  });

  it("pairwiseDupF1 is 1.0 for a perfect clustering", () => {
    expect(pairwiseDupF1([0, 0, 1, 1], [9, 9, 8, 8]).f1).toBe(1);
  });
});

describe("MLOps auto-rollback decision", () => {
  const good = { precisionAtK: 0.9, ndcgAtK: 0.9, dupF1: 0.9 };
  const bad = { precisionAtK: 0.5, ndcgAtK: 0.5, dupF1: 0.5 };

  it("evalQualityScore rewards higher metrics", () => {
    expect(evalQualityScore(good)).toBeGreaterThan(evalQualityScore(bad));
  });

  it("rolls back only on a regression beyond the threshold", () => {
    const baseline = evalQualityScore(good);
    expect(shouldRollback(evalQualityScore(bad), baseline)).toBe(true); // big drop
    expect(shouldRollback(baseline - 0.05, baseline)).toBe(false); // small drop tolerated
    expect(shouldRollback(baseline + 0.1, baseline)).toBe(false); // improvement
  });
});
