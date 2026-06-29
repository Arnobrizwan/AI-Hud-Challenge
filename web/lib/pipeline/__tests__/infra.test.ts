import { describe, it, expect } from "vitest";
import { getPrompt, listPromptVersions, renderPrompt } from "../prompts";
import { estimateTokens, estimateCostUSD, isPriced, PRICING } from "../cost";
import { assessFeedHealth } from "../../health";

describe("prompt registry", () => {
  it("getPrompt returns the ACTIVE version by default", () => {
    expect(getPrompt("summarize.system").version).toBe(2);
    expect(getPrompt("summarize.system").text).toContain("18-32 words");
  });
  it("getPrompt can pin a specific version", () => {
    expect(getPrompt("summarize.system", 1).version).toBe(1);
  });
  it("throws on unknown id or version", () => {
    expect(() => getPrompt("nope")).toThrow();
    expect(() => getPrompt("summarize.system", 99)).toThrow();
  });
  it("listPromptVersions returns retained history", () => {
    expect(listPromptVersions("summarize.system").length).toBeGreaterThanOrEqual(2);
  });
  it("renderPrompt interpolates {{vars}} and blanks missing ones", () => {
    expect(renderPrompt("summarize.user", { title: "T", text: "B" })).toBe("TITLE: T\n\nTEXT:\nB");
    expect(renderPrompt("summarize.user", { title: "T" })).toBe("TITLE: T\n\nTEXT:\n");
  });
});

describe("cost estimation", () => {
  it("estimateTokens ≈ chars/4", () => {
    expect(estimateTokens("abcd")).toBe(1);
    expect(estimateTokens("a".repeat(400))).toBe(100);
    expect(estimateTokens("")).toBe(0);
    expect(estimateTokens(null)).toBe(0);
  });
  it("estimateCostUSD prices known models by input+output", () => {
    // gpt-4o-mini: $0.15/M in, $0.60/M out
    const usd = estimateCostUSD("openai", "gpt-4o-mini", { inputTokens: 1_000_000, outputTokens: 1_000_000 });
    expect(usd).toBeCloseTo(0.75, 6);
  });
  it("embedding model has zero output cost", () => {
    const usd = estimateCostUSD("openai", "text-embedding-3-small", { inputTokens: 1_000_000, outputTokens: 5 });
    expect(usd).toBeCloseTo(0.02, 6);
  });
  it("unknown (provider, model) returns 0 and isPriced=false", () => {
    expect(estimateCostUSD("openai", "ghost-model", { inputTokens: 1000 })).toBe(0);
    expect(isPriced("openai", "ghost-model")).toBe(false);
    expect(isPriced("openai", "gpt-4o-mini")).toBe(true);
    expect(Object.keys(PRICING).length).toBeGreaterThan(0);
  });
});

describe("feed health assessment", () => {
  const now = 1_750_000_000_000;
  it("ok when 200 + items + fresh", () => {
    expect(assessFeedHealth({ httpStatus: 200, total: 50, latestPublishedAt: now - 3600_000, now }).ok).toBe(true);
  });
  it("flags non-200, empty, and stale", () => {
    expect(assessFeedHealth({ httpStatus: 502, total: 0, now }).ok).toBe(false);
    expect(assessFeedHealth({ httpStatus: 200, total: 0, now }).reasons).toContain("feed returned no items");
    const stale = assessFeedHealth({ httpStatus: 200, total: 5, latestPublishedAt: now - 100 * 3600_000, now });
    expect(stale.ok).toBe(false);
    expect(stale.reasons[0]).toMatch(/stale/);
  });
});
