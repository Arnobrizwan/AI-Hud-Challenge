import { describe, it, expect } from "vitest";
import { parseJsonFeed } from "../ingest/jsonfeed";
import { parseRobots, isPathAllowed } from "../ingest/robots";
import type { SourceSpec } from "../types";

const SOURCE: SourceSpec = {
  sourceId: "jf",
  name: "JF Source",
  kind: "jsonfeed",
  url: "https://example.com/feed.json",
  topics: ["ai"],
  weight: 0.7,
};

describe("parseJsonFeed (jsonfeed.org spec)", () => {
  it("maps items[] fields into RawItem[]", () => {
    const doc = {
      version: "https://jsonfeed.org/version/1.1",
      title: "Example",
      items: [
        {
          id: "1",
          url: "https://example.com/a",
          title: "First Post",
          content_html: "<p>Hello <b>world</b></p>",
          date_published: "2026-06-01T12:00:00Z",
          authors: [{ name: "Ada" }],
          image: "https://example.com/a.png",
        },
      ],
    };
    const items = parseJsonFeed(doc, SOURCE);
    expect(items).toHaveLength(1);
    const it = items[0];
    expect(it.title).toBe("First Post");
    expect(it.url).toBe("https://example.com/a");
    expect(it.contentText).toBe("Hello world");
    expect(it.author).toBe("Ada");
    expect(it.image).toBe("https://example.com/a.png");
    expect(it.publishedAt).toBe(Date.parse("2026-06-01T12:00:00Z"));
    expect(it.sourceId).toBe("jf");
    expect(it.sourceTopics).toEqual(["ai"]);
  });

  it("synthesizes a title for title-less microblog items", () => {
    const doc = {
      items: [
        { id: "2", url: "https://example.com/b", content_text: "just a quick thought about agents and tools" },
      ],
    };
    const items = parseJsonFeed(doc, SOURCE);
    expect(items).toHaveLength(1);
    expect(items[0].title.length).toBeGreaterThan(0);
    expect(items[0].title).toContain("just a quick");
  });

  it("falls back to external_url / id and the v1 author object", () => {
    const doc = {
      items: [
        { id: "https://example.com/c", title: "C", content_text: "c", author: { name: "Grace" } },
        { id: "x", external_url: "https://example.com/d", title: "D", content_text: "d" },
      ],
    };
    const items = parseJsonFeed(doc, SOURCE);
    expect(items.map((i) => i.url)).toEqual(["https://example.com/c", "https://example.com/d"]);
    expect(items[0].author).toBe("Grace");
  });

  it("skips items without a usable url", () => {
    const doc = { items: [{ id: "no-url", title: "Orphan", content_text: "x" }] };
    expect(parseJsonFeed(doc, SOURCE)).toHaveLength(0);
  });

  it("tolerates a missing/empty items array", () => {
    expect(parseJsonFeed({}, SOURCE)).toEqual([]);
    expect(parseJsonFeed({ items: [] }, SOURCE)).toEqual([]);
  });
});

describe("robots.txt parsing + enforcement", () => {
  const ROBOTS = `
User-agent: *
Disallow: /private
Disallow: /tmp
Crawl-delay: 2

User-agent: HUD-NewsBot
Disallow: /no-bots
Allow: /no-bots/feed.json
`;

  it("selects the specific UA group over '*'", () => {
    const r = parseRobots(ROBOTS, "hud-newsbot");
    expect(r.fetchedOk).toBe(true);
    expect(r.disallow).toContain("/no-bots");
    expect(r.allow).toContain("/no-bots/feed.json");
    // '*' group's /private should NOT apply once a specific group matched
    expect(r.disallow).not.toContain("/private");
  });

  it("falls back to '*' group for unknown agents", () => {
    const r = parseRobots(ROBOTS, "some-other-bot");
    expect(r.disallow).toContain("/private");
    expect(r.crawlDelaySec).toBe(2);
  });

  it("longest-match precedence; Allow beats Disallow on ties", () => {
    const r = parseRobots(ROBOTS, "hud-newsbot");
    expect(isPathAllowed(r, "/no-bots/page")).toBe(false); // disallowed
    expect(isPathAllowed(r, "/no-bots/feed.json")).toBe(true); // allow override (longer)
    expect(isPathAllowed(r, "/public")).toBe(true); // unmatched → allowed
  });

  it("supports * wildcard and $ end-anchor", () => {
    const r = parseRobots("User-agent: *\nDisallow: /*.pdf$\n", "hud-newsbot");
    expect(isPathAllowed(r, "/docs/report.pdf")).toBe(false);
    expect(isPathAllowed(r, "/docs/report.pdf?x=1")).toBe(true); // $ anchors end
  });

  it("fail-open when robots could not be fetched", () => {
    expect(isPathAllowed({ allow: [], disallow: [], fetchedOk: false }, "/anything")).toBe(true);
  });

  it("empty Disallow means allow-all", () => {
    const r = parseRobots("User-agent: *\nDisallow:\n", "hud-newsbot");
    expect(r.disallow).toEqual([]);
    expect(isPathAllowed(r, "/anything")).toBe(true);
  });
});
