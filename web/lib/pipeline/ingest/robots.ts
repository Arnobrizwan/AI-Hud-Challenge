/**
 * robots.txt parsing + enforcement (politeness layer for Stage 1 ingest).
 *
 * Pure functions (`parseRobots`, `isPathAllowed`) are unit-tested; `getRobots`
 * adds a cached network fetch. We honor the most specific matching User-agent
 * group, Google-style longest-match precedence (Allow wins ties), and
 * Crawl-delay. A failed/absent robots.txt defaults to "allow" (fail-open).
 */

const UA_TOKEN = "hud-newsbot"; // matches the bot's User-Agent product token

export interface RobotsRules {
  allow: string[];
  disallow: string[];
  crawlDelaySec?: number;
  /** false when robots.txt couldn't be fetched/parsed → caller should fail-open. */
  fetchedOk: boolean;
}

interface Group {
  agents: string[];
  allow: string[];
  disallow: string[];
  crawlDelaySec?: number;
}

/** Parse robots.txt text, resolving the rules that apply to `uaToken`. */
export function parseRobots(txt: string, uaToken = UA_TOKEN): RobotsRules {
  const groups: Group[] = [];
  let cur: Group | null = null;
  let lastWasAgent = false;

  for (const rawLine of txt.split(/\r?\n/)) {
    const line = rawLine.replace(/#.*$/, "").trim();
    if (!line) continue;
    const idx = line.indexOf(":");
    if (idx === -1) continue;
    const field = line.slice(0, idx).trim().toLowerCase();
    const value = line.slice(idx + 1).trim();

    if (field === "user-agent") {
      // A User-agent after a rule line starts a fresh group.
      if (!cur || !lastWasAgent) {
        cur = { agents: [], allow: [], disallow: [] };
        groups.push(cur);
      }
      cur.agents.push(value.toLowerCase());
      lastWasAgent = true;
      continue;
    }
    if (!cur) continue; // rule before any User-agent → ignore
    lastWasAgent = false;
    if (field === "disallow") cur.disallow.push(value);
    else if (field === "allow") cur.allow.push(value);
    else if (field === "crawl-delay") {
      const n = Number(value);
      if (Number.isFinite(n) && n >= 0) cur.crawlDelaySec = n;
    }
  }

  const applies = (g: Group, token: string) =>
    g.agents.some((a) => a !== "*" && (token.includes(a) || a.includes(token)));
  const specific = groups.filter((g) => applies(g, uaToken));
  const star = groups.filter((g) => g.agents.includes("*"));
  const chosen = specific.length ? specific : star;

  const merged: RobotsRules = { allow: [], disallow: [], fetchedOk: true };
  for (const g of chosen) {
    // a bare "Disallow:" (empty) means "allow all" → not a real rule
    merged.allow.push(...g.allow.filter(Boolean));
    merged.disallow.push(...g.disallow.filter(Boolean));
    if (g.crawlDelaySec != null && (merged.crawlDelaySec == null || g.crawlDelaySec > merged.crawlDelaySec))
      merged.crawlDelaySec = g.crawlDelaySec;
  }
  return merged;
}

/** Convert a robots path pattern (with `*` and `$`) to a prefix-anchored RegExp. */
function patternToRegExp(pattern: string): RegExp {
  let re = "";
  for (const ch of pattern) {
    if (ch === "*") re += ".*";
    else if (ch === "$") re += "$";
    else re += ch.replace(/[.+?^${}()|[\]\\]/g, "\\$&");
  }
  return new RegExp("^" + re);
}

function ruleMatchLen(pattern: string, path: string): number {
  // length used for precedence ignores wildcards/anchors
  return patternToRegExp(pattern).test(path) ? pattern.replace(/[*$]/g, "").length : -1;
}

/** Google-style decision: longest matching rule wins; Allow beats Disallow on ties. */
export function isPathAllowed(rules: RobotsRules, path: string): boolean {
  if (!rules.fetchedOk) return true; // fail-open
  let bestAllow = -1;
  let bestDisallow = -1;
  for (const p of rules.allow) bestAllow = Math.max(bestAllow, ruleMatchLen(p, path));
  for (const p of rules.disallow) bestDisallow = Math.max(bestDisallow, ruleMatchLen(p, path));
  if (bestDisallow < 0) return true;
  return bestAllow >= bestDisallow;
}

/** Fetch + cache robots.txt per origin (fail-open on any error). */
export async function getRobots(
  cache: Map<string, RobotsRules>,
  origin: string,
  uaToken = UA_TOKEN,
): Promise<RobotsRules> {
  const hit = cache.get(origin);
  if (hit) return hit;
  let rules: RobotsRules = { allow: [], disallow: [], fetchedOk: false };
  try {
    const res = await fetch(origin + "/robots.txt", {
      headers: { "User-Agent": uaToken },
      redirect: "follow",
    });
    if (res.ok) {
      rules = parseRobots(await res.text(), uaToken);
    } else if (res.status >= 400 && res.status < 500) {
      // 4xx (incl. 404) → no restrictions; treat as fetched-ok, allow all.
      rules = { allow: [], disallow: [], fetchedOk: true };
    }
  } catch {
    // network error → fail-open
  }
  cache.set(origin, rules);
  return rules;
}
