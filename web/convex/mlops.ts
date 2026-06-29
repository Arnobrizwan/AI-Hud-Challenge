import { mutation, query, internalMutation, type MutationCtx } from "./_generated/server";
import { v } from "convex/values";
import { requireAdmin } from "./authz";
import { DEFAULT_CONFIG } from "./defaults";
import { assignArm } from "../lib/pipeline/experiment";
import { estimateTokens, estimateCostUSD } from "../lib/pipeline/cost";

/**
 * Roll the live config back to a saved version and mark it the promoted one in
 * the registry. Shared by the manual `rollbackTo` mutation and the automated
 * eval-regression rollback. Returns false if the version doesn't exist.
 */
export async function performRollback(ctx: MutationCtx, version: number): Promise<boolean> {
  const ver = await ctx.db
    .query("configVersions")
    .withIndex("by_version", (q) => q.eq("version", version))
    .unique();
  if (!ver) return false;
  const cfg = await ctx.db.query("pipelineConfig").withIndex("by_key", (q) => q.eq("key", "default")).unique();
  const next = {
    key: "default",
    weights: ver.weights,
    recencyHalfLifeHours: ver.recencyHalfLifeHours,
    breakingVelocityThreshold: ver.breakingVelocityThreshold,
    explorationEpsilon: ver.explorationEpsilon,
    maxPerSourcePerScreen: ver.maxPerSourcePerScreen,
    updatedAt: Date.now(),
    version,
  };
  if (cfg) await ctx.db.patch(cfg._id, next);
  else await ctx.db.insert("pipelineConfig", next);
  // Keep the registry's promoted flag consistent: exactly one promoted version.
  const all = await ctx.db.query("configVersions").withIndex("by_version").order("desc").take(50);
  for (const cv of all) {
    const promoted = cv.version === version;
    if (cv.promoted !== promoted) await ctx.db.patch(cv._id, { promoted });
  }
  return true;
}

/**
 * MLOps + safety ops (sections 11, 12, 15):
 *  - config registry: versioned ranking config with promote/rollback (canary).
 *  - data-quality + drift checks that raise alerts.
 *  - spam auto-downgrade of low-quality sources.
 */

// ---- Config registry / versioning -----------------------------------------

/** Snapshot the live config as an immutable version (the model/config registry). */
export const snapshotVersion = mutation({
  args: { note: v.optional(v.string()) },
  handler: async (ctx, { note }) => {
    await requireAdmin(ctx);
    const cfg = await ctx.db.query("pipelineConfig").withIndex("by_key", (q) => q.eq("key", "default")).unique();
    const base = cfg ?? { ...DEFAULT_CONFIG };
    const last = await ctx.db.query("configVersions").withIndex("by_version").order("desc").first();
    const version = (last?.version ?? 0) + 1;
    await ctx.db.insert("configVersions", {
      version,
      weights: base.weights,
      recencyHalfLifeHours: base.recencyHalfLifeHours,
      breakingVelocityThreshold: base.breakingVelocityThreshold,
      explorationEpsilon: base.explorationEpsilon,
      maxPerSourcePerScreen: base.maxPerSourcePerScreen,
      createdAt: Date.now(),
      note,
      promoted: true,
    });
    if (cfg) await ctx.db.patch(cfg._id, { version });
    return { version };
  },
});

export const listVersions = query({
  args: {},
  handler: async (ctx) =>
    ctx.db.query("configVersions").withIndex("by_version").order("desc").take(20),
});

/** Rollback: copy a prior version's config into the live config (one-click). */
export const rollbackTo = mutation({
  args: { version: v.number() },
  handler: async (ctx, { version }) => {
    await requireAdmin(ctx);
    const ok = await performRollback(ctx, version);
    if (!ok) throw new Error("version not found");
    return { rolledBackTo: version };
  },
});

// ---- Alerts (observability) ------------------------------------------------

export const raiseAlert = internalMutation({
  args: {
    type: v.string(),
    severity: v.union(v.literal("info"), v.literal("warn"), v.literal("critical")),
    message: v.string(),
  },
  handler: async (ctx, a) => {
    await ctx.db.insert("alerts", { ...a, createdAt: Date.now(), resolved: false });
  },
});

export const listAlerts = query({
  args: {},
  handler: async (ctx) =>
    ctx.db.query("alerts").withIndex("by_createdAt").order("desc").take(25),
});

/**
 * Estimated AI cost over a recent window (cost-tracking observability).
 * Sums dense-embedding cost (items with a vector) + abstractive-summary cost
 * (items with summaryAbstractive). Token counts are approximated from text and
 * priced via lib/pipeline/cost.ts — values are ESTIMATES, not billed amounts.
 */
export const costSummary = query({
  args: { windowHours: v.optional(v.number()) },
  handler: async (ctx, { windowHours }) => {
    const hours = windowHours ?? 48;
    const cutoff = Date.now() - hours * 3600 * 1000;
    const items = await ctx.db
      .query("items")
      .withIndex("by_publishedAt", (q) => q.gte("publishedAt", cutoff))
      .take(2000);

    let embCount = 0, embUSD = 0, sumCount = 0, sumUSD = 0;
    for (const it of items) {
      if (it.vector && it.vector.length > 0) {
        embCount++;
        const inTok = estimateTokens(it.readableText ?? it.summaryExtractive ?? it.title);
        embUSD += estimateCostUSD("openai", "text-embedding-3-small", { inputTokens: inTok });
      }
      if (it.summaryAbstractive) {
        sumCount++;
        const inTok = estimateTokens(it.readableText ?? it.summaryExtractive ?? it.title);
        const outTok = estimateTokens(it.summaryAbstractive);
        sumUSD += estimateCostUSD("openai", "gpt-4o-mini", { inputTokens: inTok, outputTokens: outTok });
      }
    }
    return {
      windowHours: hours,
      items: items.length,
      embeddings: { count: embCount, usd: embUSD },
      summaries: { count: sumCount, usd: sumUSD },
      totalUSD: embUSD + sumUSD,
      note: "estimated (token≈chars/4, list prices); assumes openai models",
    };
  },
});

export const resolveAlert = mutation({
  args: { id: v.id("alerts") },
  handler: async (ctx, { id }) => {
    await requireAdmin(ctx);
    await ctx.db.patch(id, { resolved: true });
  },
});

// ---- Data-quality, drift, spam (run by the pipeline cron) ------------------

/** Schema/quality checks on recent items; raises an alert if thresholds breached. */
export const dataQualityCheck = internalMutation({
  args: {},
  handler: async (ctx) => {
    const cutoff = Date.now() - 24 * 3600 * 1000;
    const items = await ctx.db
      .query("items")
      .withIndex("by_publishedAt", (q) => q.gte("publishedAt", cutoff))
      .take(1000);
    if (items.length === 0) return { ok: true };
    const emptyTitle = items.filter((i) => !i.title.trim()).length / items.length;
    const noTopics = items.filter((i) => i.topics.length === 0).length / items.length;
    const nonEn = items.filter((i) => i.lang !== "en").length / items.length;
    if (emptyTitle > 0.02 || noTopics > 0.4 || nonEn > 0.3) {
      await ctx.db.insert("alerts", {
        type: "data_quality",
        severity: "warn",
        message: `quality: emptyTitle ${(emptyTitle * 100).toFixed(0)}% · noTopics ${(noTopics * 100).toFixed(0)}% · nonEn ${(nonEn * 100).toFixed(0)}%`,
        createdAt: Date.now(),
        resolved: false,
      });
    }
    return { ok: true };
  },
});

/** Topic-distribution drift vs the previous snapshot (Jensen-Shannon). */
export const driftCheck = internalMutation({
  args: {},
  handler: async (ctx) => {
    const cutoff = Date.now() - 12 * 3600 * 1000;
    const items = await ctx.db
      .query("items")
      .withIndex("by_publishedAt", (q) => q.gte("publishedAt", cutoff))
      .take(1500);
    if (items.length < 20) return { divergence: 0 };
    const counts = new Map<string, number>();
    let total = 0;
    for (const it of items) for (const t of it.topics) { counts.set(t, (counts.get(t) ?? 0) + 1); total++; }
    const dist = Array.from(counts.entries()).map(([topic, c]) => ({ topic, share: c / total }));
    const prev = await ctx.db.query("driftSnapshots").withIndex("by_createdAt").order("desc").first();
    let divergence = 0;
    if (prev) {
      const p = new Map(prev.topicDist.map((d) => [d.topic, d.share]));
      const q = new Map(dist.map((d) => [d.topic, d.share]));
      const keys = new Set([...p.keys(), ...q.keys()]);
      for (const k of keys) {
        const a = p.get(k) ?? 1e-6, b = q.get(k) ?? 1e-6, m = (a + b) / 2;
        divergence += 0.5 * a * Math.log(a / m) + 0.5 * b * Math.log(b / m);
      }
    }
    await ctx.db.insert("driftSnapshots", { createdAt: Date.now(), topicDist: dist, divergence });
    if (divergence > 0.15) {
      await ctx.db.insert("alerts", {
        type: "drift", severity: "warn",
        message: `topic drift JS=${divergence.toFixed(3)} (distribution shifted)`,
        createdAt: Date.now(), resolved: false,
      });
    }
    return { divergence };
  },
});

export const driftHistory = query({
  args: {},
  handler: async (ctx) =>
    ctx.db.query("driftSnapshots").withIndex("by_createdAt").order("desc").take(12),
});

// ---- A/B canary on ranking config -----------------------------------------

export const startCanary = mutation({
  args: { name: v.string(), variantVersion: v.number(), trafficPct: v.number() },
  handler: async (ctx, { name, variantVersion, trafficPct }) => {
    await requireAdmin(ctx);
    const cfg = await ctx.db.query("pipelineConfig").withIndex("by_key", (q) => q.eq("key", "default")).unique();
    await ctx.db.insert("experiments", {
      name,
      status: "running",
      controlVersion: cfg?.version ?? 0,
      variantVersion,
      trafficPct: Math.max(0, Math.min(100, trafficPct)),
      createdAt: Date.now(),
    });
  },
});

export const stopCanary = mutation({
  args: { id: v.id("experiments") },
  handler: async (ctx, { id }) => {
    await requireAdmin(ctx);
    await ctx.db.patch(id, { status: "stopped" });
  },
});

export const listExperiments = query({
  args: {},
  handler: async (ctx) => ctx.db.query("experiments").withIndex("by_status").order("desc").take(10),
});

/**
 * Attribute recent feedback to each running experiment's arms (same
 * deterministic `assignArm` routing the feed uses) and record control vs
 * variant satisfaction + sample counts so the console can show the lift.
 * Run by the pipeline cron.
 */
export const recomputeExperimentMetrics = internalMutation({
  args: {},
  handler: async (ctx) => {
    const running = await ctx.db
      .query("experiments")
      .withIndex("by_status", (q) => q.eq("status", "running"))
      .collect();
    if (running.length === 0) return { updated: 0 };

    const since = Date.now() - 7 * 24 * 3600 * 1000;
    const fb = await ctx.db.query("feedback").withIndex("by_user").order("desc").take(5000);
    const recent = fb.filter((f) => f.createdAt >= since);
    const POS = new Set(["up", "save", "more_like_this", "click"]);
    const NEG = new Set(["down", "not_interested", "mute_source"]);

    let updated = 0;
    for (const exp of running) {
      let cPos = 0, cNeg = 0, cN = 0, vPos = 0, vNeg = 0, vN = 0;
      for (const f of recent) {
        const { arm } = assignArm(String(f.userId), exp);
        const pos = POS.has(f.action), neg = NEG.has(f.action);
        if (arm === "control") { cN++; if (pos) cPos++; if (neg) cNeg++; }
        else { vN++; if (pos) vPos++; if (neg) vNeg++; }
      }
      await ctx.db.patch(exp._id, {
        metrics: {
          controlSatisfaction: cPos + cNeg > 0 ? cPos / (cPos + cNeg) : 0,
          variantSatisfaction: vPos + vNeg > 0 ? vPos / (vPos + vNeg) : 0,
          samples: cN + vN, // impressions attributed across both arms
        },
      });
      updated++;
    }
    return { updated };
  },
});

/** Auto-downgrade sources that keep failing or get muted a lot (spam plumbing). */
export const autoDowngradeSources = internalMutation({
  args: {},
  handler: async (ctx) => {
    const sources = await ctx.db.query("sources").collect();
    let changed = 0;
    for (const s of sources) {
      const tot = s.successCount + s.errorCount;
      const errRate = tot > 0 ? s.errorCount / tot : 0;
      const stat = await ctx.db
        .query("sourceStats")
        .withIndex("by_sourceId", (q) => q.eq("sourceId", s.sourceId))
        .unique();
      const muteRate = stat?.muteRate ?? 0;
      const spamScore = Math.min(1, 0.5 * errRate + 0.5 * muteRate);
      const autoDowngraded = spamScore > 0.5;
      if (s.spamScore !== spamScore || s.autoDowngraded !== autoDowngraded) {
        await ctx.db.patch(s._id, {
          spamScore,
          qualityScore: stat?.satisfaction,
          autoDowngraded,
        });
        changed++;
      }
    }
    return { changed };
  },
});
