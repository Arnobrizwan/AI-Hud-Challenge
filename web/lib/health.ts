/**
 * Health assessment for the deployed feed (used by scripts/healthcheck.mjs).
 * Pure + unit-tested; the script mirrors this logic for a dependency-free run.
 */

export interface FeedHealthInput {
  httpStatus: number;
  total?: number;
  latestPublishedAt?: number; // max publishedAt across returned items (ms)
  now: number;
  maxStalenessHours?: number; // default 72 (matches the feed window)
}

export interface HealthResult {
  ok: boolean;
  reasons: string[];
}

export function assessFeedHealth(input: FeedHealthInput): HealthResult {
  const reasons: string[] = [];
  if (input.httpStatus !== 200) reasons.push(`http ${input.httpStatus}`);
  if ((input.total ?? 0) <= 0) reasons.push("feed returned no items");
  const maxMs = (input.maxStalenessHours ?? 72) * 3600 * 1000;
  if (input.latestPublishedAt != null && input.now - input.latestPublishedAt > maxMs) {
    const ageH = Math.round((input.now - input.latestPublishedAt) / 3_600_000);
    reasons.push(`stale: newest item ${ageH}h old (> ${input.maxStalenessHours ?? 72}h)`);
  }
  return { ok: reasons.length === 0, reasons };
}
