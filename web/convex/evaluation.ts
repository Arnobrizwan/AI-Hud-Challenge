import { query, mutation } from "./_generated/server";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";
import { DEFAULT_PREFS, DEFAULT_CONFIG } from "./defaults";
import { scoreForUser, topicalMatch, type UserContext } from "../lib/pipeline/rank";
import { normalizeTitle } from "../lib/pipeline/text";
import { isGrounded } from "../lib/pipeline/summarize";

const WINDOW_MS = 48 * 3600 * 1000;

/**
 * Offline evaluation harness. Computes Precision@K, nDCG@K, coverage, novelty,
 * diversity over the live ranking, plus a dedupe-quality proxy from clustering.
 * Relevance = explicit feedback when present, else a topical-match proxy.
 */
export const runEval = mutation({
  args: { k: v.optional(v.number()) },
  handler: async (ctx, { k }) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    const K = k ?? 10;

    const cutoff = Date.now() - WINDOW_MS;
    const all = await ctx.db
      .query("items")
      .withIndex("by_publishedAt", (q) => q.gte("publishedAt", cutoff))
      .collect();
    const items = all.filter((i) => i.isRepresentative);

    const prefs = await ctx.db
      .query("userPrefs")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .unique();
    const cfg = await ctx.db
      .query("pipelineConfig")
      .withIndex("by_key", (q) => q.eq("key", "default"))
      .unique();
    const weights = cfg?.weights ?? DEFAULT_CONFIG.weights;

    const fb = await ctx.db
      .query("feedback")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .collect();
    const rel = new Map<string, number>();
    for (const f of fb) {
      if (f.action === "up") rel.set(f.itemId, 1);
      if (f.action === "down" || f.action === "not_interested") rel.set(f.itemId, 0);
    }

    const focusTopics = prefs?.focusTopics ?? DEFAULT_PREFS.focusTopics;
    const userCtx: UserContext = {
      focusTopics,
      boostedSources: prefs?.boostedSources ?? DEFAULT_PREFS.boostedSources,
      mutedSources: prefs?.mutedSources ?? DEFAULT_PREFS.mutedSources,
      focusVsPopularMix: prefs?.focusVsPopularMix ?? DEFAULT_PREFS.focusVsPopularMix,
      seen: new Set(),
    };

    const ranked = items
      .map((it) => ({
        it,
        s: scoreForUser(
          { topics: it.topics, sourceId: it.sourceId, id: it._id, features: it.features },
          userCtx,
          weights,
        ),
      }))
      .sort((a, b) => b.s.score - a.s.score);

    // Evaluate the SAME per-source-diversified list the feed serves, so coverage
    // and diversity reflect what the user actually sees (not the raw ranking).
    const maxPerSource = cfg?.maxPerSourcePerScreen ?? DEFAULT_CONFIG.maxPerSourcePerScreen;
    const perSource = new Map<string, number>();
    const diversified = ranked.filter((r) => {
      const n = perSource.get(r.it.sourceId) ?? 0;
      if (n >= maxPerSource) return false;
      perSource.set(r.it.sourceId, n + 1);
      return true;
    });
    const topK = diversified.slice(0, K);

    // relevance: explicit feedback else topical-match proxy (>=0.34 → relevant)
    const relevance = (itemId: string, topics: string[], sourceId: string): number => {
      if (rel.has(itemId)) return rel.get(itemId)!;
      return topicalMatch(topics, focusTopics, sourceId, userCtx.boostedSources) >= 0.34 ? 1 : 0;
    };

    let hits = 0;
    let dcg = 0;
    let idealRels: number[] = [];
    topK.forEach((r, i) => {
      const g = relevance(r.it._id, r.it.topics, r.it.sourceId);
      if (g > 0) hits++;
      dcg += g / Math.log2(i + 2);
      idealRels.push(g);
    });
    idealRels = idealRels.sort((a, b) => b - a);
    const idcg = idealRels.reduce((acc, g, i) => acc + g / Math.log2(i + 2), 0);

    const precisionAtK = topK.length ? hits / topK.length : 0;
    const ndcgAtK = idcg > 0 ? dcg / idcg : 0;

    const distinctSources = new Set(topK.map((r) => r.it.sourceId));
    const enabledSources = (await ctx.db.query("sources").collect()).filter((s) => s.enabled);
    const coverage = enabledSources.length ? distinctSources.size / enabledSources.length : 0;
    const diversity = topK.length ? distinctSources.size / topK.length : 0;
    const novelty =
      topK.length
        ? topK.reduce((acc, r) => acc + (1 - r.it.features.popularity), 0) / topK.length
        : 0;

    // dedupe-quality proxy: of title-collision pairs, fraction grouped together.
    const clusters = await ctx.db
      .query("clusters")
      .withIndex("by_lastUpdated", (q) => q.gte("lastUpdatedAt", cutoff))
      .collect();
    const byTitle = new Map<string, string[]>();
    for (const it of items) {
      const key = normalizeTitle(it.title).split(" ").slice(0, 6).join(" ");
      const arr = byTitle.get(key);
      if (arr) arr.push(String(it.clusterId ?? it._id));
      else byTitle.set(key, [String(it.clusterId ?? it._id)]);
    }
    let pairsTotal = 0;
    let pairsGrouped = 0;
    for (const cl of byTitle.values()) {
      if (cl.length < 2) continue;
      for (let i = 0; i < cl.length; i++)
        for (let j = i + 1; j < cl.length; j++) {
          pairsTotal++;
          if (cl[i] === cl[j]) pairsGrouped++;
        }
    }
    const dupRecall = pairsTotal ? pairsGrouped / pairsTotal : 1;
    const multiSourceRatio = clusters.length
      ? clusters.filter((c) => c.memberCount > 1).length / clusters.length
      : 0;
    const dupF1 = (2 * dupRecall * (multiSourceRatio || 1)) / (dupRecall + (multiSourceRatio || 1) || 1);

    // cluster purity: for multi-member clusters, share whose members agree on the
    // cluster's dominant topic.
    let purSum = 0, purN = 0;
    for (const c of clusters) {
      if (c.memberCount < 2) continue;
      const members = await ctx.db
        .query("items")
        .withIndex("by_cluster", (q) => q.eq("clusterId", c._id))
        .collect();
      if (members.length < 2) continue;
      const counts = new Map<string, number>();
      for (const m of members) for (const t of m.topics) counts.set(t, (counts.get(t) ?? 0) + 1);
      const top = Math.max(0, ...counts.values());
      purSum += top / members.length;
      purN++;
    }
    const clusterPurity = purN ? purSum / purN : 1;

    // summary factuality: fraction of abstractive summaries that pass grounding.
    const withAbs = items.filter((i) => i.summaryAbstractive);
    let grounded = 0;
    for (const i of withAbs) {
      if (isGrounded(i.summaryAbstractive!, (i.readableText ?? i.summaryExtractive) + " " + i.title))
        grounded++;
    }
    const factuality = withAbs.length ? grounded / withAbs.length : 1;

    // time-to-surface: median (fetchedAt - publishedAt) over the window.
    const lags = items.map((i) => Math.max(0, i.fetchedAt - i.publishedAt)).sort((a, b) => a - b);
    const timeToSurfaceMs = lags.length ? lags[Math.floor(lags.length / 2)] : 0;

    // gold-set precision (if a curated gold set exists): top-K titles matching a gold keyword.
    const gold = await ctx.db.query("goldSet").collect();
    let goldNote = "";
    if (gold.length > 0) {
      const hit = topK.filter((r) =>
        gold.some((g) => r.it.title.toLowerCase().includes(g.keyword.toLowerCase())),
      ).length;
      goldNote = ` · gold-hit ${hit}/${topK.length}`;
    }

    const metrics = {
      precisionAtK,
      ndcgAtK,
      coverage,
      novelty,
      dupF1,
      diversity,
      clusterPurity,
      factuality,
      timeToSurfaceMs,
    };

    const cfgVersion = cfg?.version;
    await ctx.db.insert("evalRuns", {
      createdAt: Date.now(),
      k: K,
      metrics,
      sampleSize: items.length,
      notes: (rel.size > 0 ? "explicit+proxy relevance" : "proxy relevance") + goldNote,
      configVersion: cfgVersion,
    });

    return { metrics, sampleSize: items.length };
  },
});

export const listEvals = query({
  args: {},
  handler: async (ctx) => {
    return await ctx.db
      .query("evalRuns")
      .withIndex("by_createdAt")
      .order("desc")
      .take(20);
  },
});
