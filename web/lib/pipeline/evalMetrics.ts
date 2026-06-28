/**
 * Pure ranking/dedup evaluation metrics. Shared by the live Convex eval
 * (`convex/evaluation.ts`) and the offline CI gate
 * (`__tests__/eval.gate.test.ts`) so the gate enforces the *same* math the
 * dashboard reports.
 */

/** Fraction of the top-K items that are relevant (graded relevance > 0). */
export function precisionAtK(relsInRankedOrder: number[], k = 10): number {
  const top = relsInRankedOrder.slice(0, k);
  if (!top.length) return 0;
  return top.filter((r) => r > 0).length / top.length;
}

/** nDCG@K with graded relevance; ideal ordering taken from the candidate set. */
export function ndcgAtK(relsInRankedOrder: number[], k = 10): number {
  const top = relsInRankedOrder.slice(0, k);
  const dcg = top.reduce((acc, g, i) => acc + g / Math.log2(i + 2), 0);
  const ideal = [...relsInRankedOrder].sort((a, b) => b - a).slice(0, k);
  const idcg = ideal.reduce((acc, g, i) => acc + g / Math.log2(i + 2), 0);
  return idcg > 0 ? dcg / idcg : 0;
}

export function f1(precision: number, recall: number): number {
  return precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
}

export interface GoldEntry {
  topic: string;
  keyword: string; // a title substring marking a relevant story
  relevance: number; // graded 0..1
}

/** Graded gold relevance for a title = max relevance over matching keywords. */
export function goldRelevance(title: string, gold: GoldEntry[]): number {
  const t = title.toLowerCase();
  let best = 0;
  for (const g of gold) if (t.includes(g.keyword.toLowerCase())) best = Math.max(best, g.relevance);
  return best;
}

/**
 * Pairwise clustering F1 for deduplication quality. `predicted` and `gold` are
 * cluster labels per item (same index space). Computes precision/recall/F1 over
 * all co-membership pairs — the standard dup-detection quality metric.
 */
export function pairwiseDupF1(
  predicted: Array<number | string>,
  gold: Array<number | string>,
): { precision: number; recall: number; f1: number } {
  const n = Math.min(predicted.length, gold.length);
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const samePred = predicted[i] === predicted[j];
      const sameGold = gold[i] === gold[j];
      if (samePred && sameGold) tp++;
      else if (samePred && !sameGold) fp++;
      else if (!samePred && sameGold) fn++;
    }
  }
  const precision = tp + fp > 0 ? tp / (tp + fp) : 1;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 1;
  return { precision, recall, f1: f1(precision, recall) };
}

/**
 * Enforced quality gates. CI fails if a metric drops below its floor — this
 * turns the challenge's acceptance criteria into a hard regression guard.
 */
export const EVAL_GATE_THRESHOLDS = {
  precisionAt10: 0.6,
  ndcgAt10: 0.7,
  dupF1: 0.7,
};

// ---- MLOps auto-rollback ---------------------------------------------------

export interface QualityMetrics {
  precisionAtK: number;
  ndcgAtK: number;
  dupF1: number;
}

/** Single composite quality score for a config version (higher = better). */
export function evalQualityScore(m: QualityMetrics): number {
  return 0.4 * m.precisionAtK + 0.4 * m.ndcgAtK + 0.2 * m.dupF1;
}

/** A new config auto-rolls back if its quality drops this far below the baseline. */
export const ROLLBACK_REGRESSION_THRESHOLD = 0.1;

/** True when the candidate config regressed beyond the threshold vs baseline. */
export function shouldRollback(
  candidate: number,
  baseline: number,
  threshold = ROLLBACK_REGRESSION_THRESHOLD,
): boolean {
  return candidate < baseline - threshold;
}
