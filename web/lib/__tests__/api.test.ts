import { describe, it, expect } from "vitest";
import {
  clampInt, parseFeedParams, notModified, bearerToken, FEED_MAX_LIMIT,
} from "../api";

describe("api helpers", () => {
  it("clampInt bounds and defaults", () => {
    expect(clampInt("5", 30, 1, 100)).toBe(5);
    expect(clampInt(null, 30, 1, 100)).toBe(30);
    expect(clampInt("abc", 30, 1, 100)).toBe(30);
    expect(clampInt("0", 30, 1, 100)).toBe(1); // below min
    expect(clampInt("999", 30, 1, 100)).toBe(100); // above max
    expect(clampInt("2.9", 30, 1, 100)).toBe(2); // truncates
  });

  it("parseFeedParams reads topic/limit/offset with caps", () => {
    const p = parseFeedParams(new URLSearchParams("topic=ai&limit=2&offset=4"));
    expect(p).toEqual({ topic: "ai", limit: 2, offset: 4 });
    const d = parseFeedParams(new URLSearchParams(""));
    expect(d).toEqual({ topic: undefined, limit: 30, offset: 0 });
    const capped = parseFeedParams(new URLSearchParams(`limit=${FEED_MAX_LIMIT + 50}`));
    expect(capped.limit).toBe(FEED_MAX_LIMIT);
  });

  it("notModified only when both present and equal", () => {
    expect(notModified('W/"1-2"', 'W/"1-2"')).toBe(true);
    expect(notModified('W/"1-2"', 'W/"9-9"')).toBe(false);
    expect(notModified(null, 'W/"1-2"')).toBe(false);
    expect(notModified('W/"1-2"', undefined)).toBe(false);
  });

  it("bearerToken strips the Bearer prefix", () => {
    expect(bearerToken("Bearer abc.def")).toBe("abc.def");
    expect(bearerToken("bearer xyz")).toBe("xyz");
    expect(bearerToken("rawtoken")).toBe("rawtoken");
    expect(bearerToken(null)).toBeUndefined();
  });
});
