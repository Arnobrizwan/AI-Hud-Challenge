import { enrichItem } from "./enrich";
import type { NormalizedItem } from "./types";

/**
 * Measured evaluation of the enrichment stage (NER + topic classification)
 * against a tiny hand-labeled set. Pure + deterministic, so it runs both in the
 * Convex eval (`evaluation.nerTopicEval`) and in unit tests.
 */

export interface NerGoldDoc {
  title: string;
  text: string;
  entities: string[]; // expected entities (case-insensitive)
  topics: string[]; // expected topics from the taxonomy
}

export interface PRF {
  precision: number;
  recall: number;
  f1: number;
}

/** Micro set precision/recall/F1 between predicted and gold string sets. */
export function setPRF(predicted: string[], gold: string[]): { tp: number; fp: number; fn: number } {
  const p = new Set(predicted.map((s) => s.toLowerCase()));
  const g = new Set(gold.map((s) => s.toLowerCase()));
  let tp = 0;
  for (const x of p) if (g.has(x)) tp++;
  return { tp, fp: p.size - tp, fn: g.size - tp };
}

function prf(tp: number, fp: number, fn: number): PRF {
  const precision = tp + fp > 0 ? tp / (tp + fp) : 1;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 1;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  return { precision, recall, f1 };
}

/** Run the enrichment stage over the gold docs and aggregate micro P/R/F1. */
export function evaluateNerTopics(docs: NerGoldDoc[]): { entity: PRF; topic: PRF; sampleSize: number } {
  let eTp = 0, eFp = 0, eFn = 0;
  let tTp = 0, tFp = 0, tFn = 0;
  for (const d of docs) {
    // Minimal NormalizedItem; sourceTopics empty so we measure the *classifier*,
    // not the source-topic floor.
    const item = {
      title: d.title,
      contentText: d.text,
      summaryExtractive: d.text,
      sourceTopics: [],
    } as unknown as NormalizedItem;
    const enriched = enrichItem(item);
    const e = setPRF(enriched.entities, d.entities);
    eTp += e.tp; eFp += e.fp; eFn += e.fn;
    const t = setPRF(enriched.topics, d.topics);
    tTp += t.tp; tFp += t.fp; tFn += t.fn;
  }
  return { entity: prf(eTp, eFp, eFn), topic: prf(tTp, tFp, tFn), sampleSize: docs.length };
}

/** Tiny hand-labeled set covering the known-org NER + keyword topic classifier. */
export const NER_TOPIC_GOLD: NerGoldDoc[] = [
  {
    title: "OpenAI releases GPT-5 with major reasoning gains",
    text: "OpenAI today announced GPT-5, its newest large language model with stronger reasoning.",
    entities: ["OpenAI"],
    topics: ["ai", "llm"],
  },
  {
    title: "Nvidia unveils new GPU for AI training",
    text: "Nvidia revealed a next-generation GPU aimed at accelerating machine learning training workloads.",
    entities: ["Nvidia"],
    topics: ["hardware", "ai", "ml"],
  },
  {
    title: "Google DeepMind publishes protein folding research",
    text: "Google DeepMind shared new research advancing scientific understanding of protein structures.",
    entities: ["Google", "DeepMind"],
    topics: ["ai", "science"],
  },
  {
    title: "Critical security vulnerability found in popular library",
    text: "A serious security vulnerability and CVE were disclosed in a widely used open source library.",
    entities: [],
    topics: ["security", "open-source"],
  },
  {
    title: "Anthropic launches Claude agent tooling for developers",
    text: "Anthropic introduced new agent tooling built on Claude with support for tool use and MCP.",
    entities: ["Anthropic"],
    topics: ["ai", "llm", "agents"],
  },
];
