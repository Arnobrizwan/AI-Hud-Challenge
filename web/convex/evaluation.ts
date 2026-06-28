import { query, mutation } from "./_generated/server";
import { v } from "convex/values";
import { requireAdmin } from "./authz";
import { DEFAULT_PREFS } from "./defaults";
import { effectiveConfig } from "./config";
import { scoreForUser, topicalMatch, type UserContext } from "../lib/pipeline/rank";
import { normalizeTitle } from "../lib/pipeline/text";
import { isGrounded } from "../lib/pipeline/summarize";
import {
  precisionAtK as precAtK, ndcgAtK as ndcgK, goldRelevance,
  evalQualityScore, shouldRollback, ROLLBACK_REGRESSION_THRESHOLD,
} from "../lib/pipeline/evalMetrics";
import { performRollback } from "./mlops";
import { evaluateNerTopics, NER_TOPIC_GOLD } from "../lib/pipeline/nerEval";

const WINDOW_MS = 48 * 3600 * 1000;

/**
 * Offline evaluation harness. Computes Precision@K, nDCG@K, coverage, novelty,
 * diversity over the live ranking, plus a dedupe-quality proxy from clustering.
 * Relevance = explicit feedback when present, else a topical-match proxy.
 */
export const runEval = mutation({
  args: { k: v.optional(v.number()) },
  handler: async (ctx, { k }) => {
    const userId = await requireAdmin(ctx);
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
    // Respect a running A/B experiment: evaluate under the config this user is
    // actually routed to (control or variant), not just the live default.
    const eff = await effectiveConfig(ctx, userId);
    const weights = eff.weights;

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
    const maxPerSource = eff.maxPerSourcePerScreen;
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

    // Relevance vector in ranked order → shared Precision@K / nDCG@K (same math
    // the offline CI gate enforces, lib/pipeline/evalMetrics.ts).
    const rels = topK.map((r) => relevance(r.it._id, r.it.topics, r.it.sourceId));
    const precisionAtK = precAtK(rels, K);
    const ndcgAtK = ndcgK(rels, K);

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

    // gold-set precision (if a curated gold set exists): graded relevance over
    // the top-K titles, via the same goldRelevance() the CI gate uses.
    const gold = await ctx.db.query("goldSet").collect();
    let goldNote = "";
    if (gold.length > 0) {
      const goldEntries = gold.map((g) => ({ topic: g.topic, keyword: g.keyword, relevance: g.relevance }));
      const goldRels = topK.map((r) => goldRelevance(r.it.title, goldEntries));
      goldNote = ` · gold-P@${K} ${precAtK(goldRels, K).toFixed(2)} · gold-nDCG ${ndcgK(goldRels, K).toFixed(2)}`;
    }

    // measured enrichment quality (NER + topic) against a tiny labeled set.
    const ner = evaluateNerTopics(NER_TOPIC_GOLD);
    const nerNote =
      ` · NER P/R ${ner.entity.precision.toFixed(2)}/${ner.entity.recall.toFixed(2)}` +
      ` · topic P/R ${ner.topic.precision.toFixed(2)}/${ner.topic.recall.toFixed(2)}`;

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

    const cfgVersion = eff.version;
    await ctx.db.insert("evalRuns", {
      createdAt: Date.now(),
      k: K,
      metrics,
      sampleSize: items.length,
      notes: (rel.size > 0 ? "explicit+proxy relevance" : "proxy relevance") + goldNote + nerNote,
      configVersion: cfgVersion,
    });

    // ---- MLOps auto-rollback (section 11): if this eval regressed beyond the
    // threshold vs the last eval under a *previous* promoted config version,
    // roll the live config back to that version and raise a critical alert. ----
    let rolledBackTo: number | null = null;
    if (cfgVersion != null) {
      const candidateScore = evalQualityScore({
        precisionAtK: metrics.precisionAtK,
        ndcgAtK: metrics.ndcgAtK,
        dupF1: metrics.dupF1,
      });
      const history = await ctx.db.query("evalRuns").withIndex("by_createdAt").order("desc").take(30);
      const baseline = history.find(
        (r) => r.configVersion != null && r.configVersion !== cfgVersion,
      );
      if (baseline && baseline.configVersion != null) {
        const baselineScore = evalQualityScore({
          precisionAtK: baseline.metrics.precisionAtK,
          ndcgAtK: baseline.metrics.ndcgAtK,
          dupF1: baseline.metrics.dupF1,
        });
        if (shouldRollback(candidateScore, baselineScore)) {
          const ok = await performRollback(ctx, baseline.configVersion);
          if (ok) {
            rolledBackTo = baseline.configVersion;
            await ctx.db.insert("alerts", {
              type: "drift",
              severity: "critical",
              message:
                `auto-rollback: eval quality ${candidateScore.toFixed(3)} for config v${cfgVersion} ` +
                `regressed >${ROLLBACK_REGRESSION_THRESHOLD} below v${baseline.configVersion} ` +
                `(${baselineScore.toFixed(3)}) — reverted to v${baseline.configVersion}`,
              createdAt: Date.now(),
              resolved: false,
            });
          }
        }
      }
    }

    return { metrics, sampleSize: items.length, rolledBackTo };
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

/**
 * Measured enrichment quality: entity (NER) + topic precision/recall/F1 against
 * the hand-labeled set. Pure — no DB read — so the dashboard can show how good
 * the classifier actually is, not just self-reported coverage.
 */
export const nerTopicEval = query({
  args: {},
  handler: async () => evaluateNerTopics(NER_TOPIC_GOLD),
});
