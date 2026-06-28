import { describe, it, expect } from "vitest";
import { userBucket, assignArm } from "../experiment";
import { denseEmbedding } from "../text";

describe("A/B canary assignment (experiment.ts)", () => {
  const exp = { controlVersion: 1, variantVersion: 2, trafficPct: 50 };

  it("userBucket is deterministic and within 0..99", () => {
    const a = userBucket("user-abc");
    expect(a).toBe(userBucket("user-abc"));
    expect(a).toBeGreaterThanOrEqual(0);
    expect(a).toBeLessThan(100);
  });

  it("trafficPct=0 → everyone control; 100 → everyone variant", () => {
    for (const u of ["a", "b", "c", "d", "e"]) {
      expect(assignArm(u, { ...exp, trafficPct: 0 })).toEqual({ arm: "control", version: 1 });
      expect(assignArm(u, { ...exp, trafficPct: 100 })).toEqual({ arm: "variant", version: 2 });
    }
  });

  it("splits roughly by trafficPct across many users", () => {
    let variant = 0;
    const N = 1000;
    for (let i = 0; i < N; i++) if (assignArm("user-" + i, exp).arm === "variant") variant++;
    expect(variant / N).toBeGreaterThan(0.4);
    expect(variant / N).toBeLessThan(0.6);
  });

  it("assignment is stable for the same user", () => {
    expect(assignArm("steady", exp)).toEqual(assignArm("steady", exp));
  });
});

describe("dense embeddings fallback (text.denseEmbedding)", () => {
  it("returns null for Anthropic (no embeddings endpoint) → caller falls back", async () => {
    expect(await denseEmbedding({ provider: "anthropic", key: "k", texts: ["x"] })).toBeNull();
  });
  it("returns null with no key or no texts (no network call)", async () => {
    expect(await denseEmbedding({ provider: "openai", key: "", texts: ["x"] })).toBeNull();
    expect(await denseEmbedding({ provider: "openai", key: "k", texts: [] })).toBeNull();
  });
});
