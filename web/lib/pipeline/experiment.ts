/**
 * Deterministic A/B canary assignment. Pure + stable so the SAME user always
 * lands in the same arm across feed reads, evaluation, and metric attribution
 * (no DB writes, no Math.random — query-safe).
 */

export interface ExperimentArms {
  controlVersion: number;
  variantVersion: number;
  trafficPct: number; // 0..100 of users routed to the variant
}

/** Stable bucket 0..99 from a user id (FNV-1a). */
export function userBucket(userId: string): number {
  let h = 2166136261;
  for (let i = 0; i < userId.length; i++) {
    h ^= userId.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) % 100;
}

/** Assign a user to control vs variant and return the config version to apply. */
export function assignArm(
  userId: string,
  exp: ExperimentArms,
): { arm: "control" | "variant"; version: number } {
  const pct = Math.max(0, Math.min(100, exp.trafficPct));
  return userBucket(userId) < pct
    ? { arm: "variant", version: exp.variantVersion }
    : { arm: "control", version: exp.controlVersion };
}
