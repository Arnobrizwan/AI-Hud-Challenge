#!/usr/bin/env node
/**
 * Post-deploy smoke test: hits the live feed API and verifies it's healthy.
 *   HEALTHCHECK_URL=https://hud-news.vercel.app node scripts/healthcheck.mjs
 * Exits 0 if healthy, 1 otherwise. Logic mirrors lib/health.ts::assessFeedHealth
 * (kept inline so the script runs with zero build/deps).
 */

const BASE = process.env.HEALTHCHECK_URL || "https://hud-news.vercel.app";
const MAX_STALE_H = Number(process.env.HEALTHCHECK_MAX_STALENESS_HOURS || 72);

function assess({ httpStatus, total, latestPublishedAt, now }) {
  const reasons = [];
  if (httpStatus !== 200) reasons.push(`http ${httpStatus}`);
  if ((total ?? 0) <= 0) reasons.push("feed returned no items");
  const maxMs = MAX_STALE_H * 3600 * 1000;
  if (latestPublishedAt != null && now - latestPublishedAt > maxMs) {
    const ageH = Math.round((now - latestPublishedAt) / 3_600_000);
    reasons.push(`stale: newest item ${ageH}h old (> ${MAX_STALE_H}h)`);
  }
  return { ok: reasons.length === 0, reasons };
}

async function main() {
  const url = `${BASE}/api/feed?limit=50`;
  let res, body;
  try {
    res = await fetch(url, { headers: { "User-Agent": "HUD-healthcheck/1.0" } });
    body = await res.json().catch(() => ({}));
  } catch (e) {
    console.error(`[healthcheck] FAIL — network error hitting ${url}: ${e.message}`);
    process.exit(1);
  }
  const items = Array.isArray(body.items) ? body.items : [];
  const latestPublishedAt = items.reduce((m, it) => Math.max(m, it.publishedAt ?? 0), 0) || undefined;
  const result = assess({ httpStatus: res.status, total: body.total ?? items.length, latestPublishedAt, now: Date.now() });

  if (result.ok) {
    console.log(`[healthcheck] OK — ${BASE} · ${body.total ?? items.length} items`);
    process.exit(0);
  }
  console.error(`[healthcheck] FAIL — ${BASE}: ${result.reasons.join("; ")}`);
  process.exit(1);
}

main();
