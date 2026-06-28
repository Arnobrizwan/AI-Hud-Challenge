import { internalAction, action } from "./_generated/server";
import { v } from "convex/values";
import { internal } from "./_generated/api";

import { ingestSource } from "../lib/pipeline/ingest";
import { getRobots, isPathAllowed, type RobotsRules } from "../lib/pipeline/ingest/robots";
import { normalizeBatch } from "../lib/pipeline/normalize";
import { enrichBatch } from "../lib/pipeline/enrich";
import { dedupCluster } from "../lib/pipeline/dedup";
import { computeItemFeatures } from "../lib/pipeline/rank";
import { simHash, hashingVector, tokenize, normalizeTitle } from "../lib/pipeline/text";
import { linkEntities } from "../lib/pipeline/kb";
import { embedTexts } from "../lib/pipeline/summarize";
import { isFlagged } from "../lib/pipeline/safety";
import type { RawItem } from "../lib/pipeline/types";
import { DEFAULT_CONFIG } from "./defaults";

const MAX_AGE_MS = 5 * 24 * 3600 * 1000; // ingest items up to 5 days old
const RECENT_WINDOW_MS = 5 * 24 * 3600 * 1000; // prune older than 5 days (> 72h feed window)

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

type Stage = { name: string; ms: number; inCount: number; outCount: number; error?: string };

/** The full ingest → enrich → dedup → rank → persist run. */
export const runPipeline = internalAction({
  args: { trigger: v.string() },
  handler: async (ctx, { trigger }): Promise<{ inserted: number; duplicates: number; clusters: number }> => {
    const runId = await ctx.runMutation(internal.pipelineStore.startRun, { trigger });
    const stages: Stage[] = [];
    const time = async <T>(name: string, inCount: number, fn: () => Promise<T> | T): Promise<T> => {
      const t0 = Date.now();
      try {
        const out = await fn();
        stages.push({ name, ms: Date.now() - t0, inCount, outCount: Array.isArray(out) ? out.length : 0 });
        return out;
      } catch (e) {
        stages.push({ name, ms: Date.now() - t0, inCount, outCount: 0, error: (e as Error).message });
        throw e;
      }
    };

    try {
      const cfg = await ctx.runQuery(internal.config.getConfigInternal, {});
      const halfLifeHours = cfg?.recencyHalfLifeHours ?? DEFAULT_CONFIG.recencyHalfLifeHours;

      const sources = await ctx.runQuery(internal.sources.listEnabled, {});

      // ---- Stage 1: ingest (sequential to be polite to sources) ----
      // robots.txt is fetched once per origin (cached) and enforced before each
      // request; Crawl-delay raises the politeness sleep. Per-domain budgets logged.
      const robotsCache = new Map<string, RobotsRules>();
      const domainBudget = new Map<string, { fetched: number; skipped: number; crawlDelaySec: number }>();
      const bumpBudget = (host: string, key: "fetched" | "skipped", crawlDelaySec = 0) => {
        const b = domainBudget.get(host) ?? { fetched: 0, skipped: 0, crawlDelaySec: 0 };
        b[key]++;
        if (crawlDelaySec) b.crawlDelaySec = crawlDelaySec;
        domainBudget.set(host, b);
      };

      const raws: RawItem[] = await time("ingest", sources.length, async () => {
        const all: RawItem[] = [];
        for (const s of sources) {
          // be polite: stagger requests, extra spacing for rate-limited hosts
          let baseDelay = s.kind === "reddit" || s.kind === "x" ? 700 : 250;

          // robots.txt enforcement (only for real http(s) feed urls)
          let httpHost: string | null = null;
          let path = "/";
          if (/^https?:\/\//i.test(s.url)) {
            try {
              const u = new URL(s.url);
              httpHost = u.host;
              path = (u.pathname || "/") + (u.search || "");
              const robots = await getRobots(robotsCache, u.origin);
              if (robots.crawlDelaySec) baseDelay = Math.max(baseDelay, robots.crawlDelaySec * 1000);
              if (!isPathAllowed(robots, path)) {
                bumpBudget(httpHost, "skipped", robots.crawlDelaySec ?? 0);
                await ctx.runMutation(internal.sources.recordFetch, {
                  sourceId: s.sourceId,
                  ok: false,
                  error: `robots.txt disallows ${path}`,
                });
                continue;
              }
            } catch {
              httpHost = null; // malformed url → skip robots, let adapter surface the error
            }
          }

          await sleep(baseDelay);
          const res = await ingestSource({
            sourceId: s.sourceId,
            name: s.name,
            kind: s.kind,
            url: s.url,
            topics: s.topics,
            weight: s.weight,
            etag: s.etag,
            lastModified: s.lastModified,
          });
          await ctx.runMutation(internal.sources.recordFetch, {
            sourceId: s.sourceId,
            ok: !res.error,
            etag: res.etag,
            lastModified: res.lastModified,
            error: res.error,
          });
          if (httpHost) bumpBudget(httpHost, "fetched");
          if (res.items.length) all.push(...res.items);
        }
        for (const [h, b] of domainBudget) {
          console.log(
            `[robots] ${h}: fetched=${b.fetched} skipped=${b.skipped} crawlDelay=${b.crawlDelaySec}s`,
          );
        }
        return all;
      });

      const fresh = raws.filter(
        (r) => !r.publishedAt || Date.now() - r.publishedAt < MAX_AGE_MS,
      );

      // ---- Stage 2-3: normalize + enrich ----
      const normalized = await time("normalize", fresh.length, () => normalizeBatch(fresh));
      const enEn = normalized.filter((n) => n.lang === "en");
      const enriched = await time("enrich", enEn.length, () => enrichBatch(enEn));

      // ---- Stage 4: dedup / cluster ----
      const dedup = await time("dedup", enriched.length, () => dedupCluster(enriched));

      // ---- Stage 5: features (velocity proxy = cluster size) ----
      const velocityByIndex = enriched.map((_, i) => {
        const c = dedup.clusters[dedup.itemCluster[i]];
        return c ? c.memberIndexes.length : 1;
      });
      const features = computeItemFeatures(enriched, { halfLifeHours, velocityByIndex });

      // assemble cluster payloads
      const clusterPayloads = dedup.clusters.map((c) => {
        const memberTopics = new Set<string>();
        let pop = 0;
        for (const mi of c.memberIndexes) {
          enriched[mi].topics.forEach((t) => memberTopics.add(t));
          pop = Math.max(pop, features[mi].popularity);
        }
        return {
          title: enriched[c.representativeIndex].title,
          memberCount: c.memberIndexes.length,
          topics: Array.from(memberTopics).slice(0, 5),
          velocity: features[c.representativeIndex].velocity,
          popularity: pop,
        };
      });

      // ---- Stage 3b: SimHash + semantic vector + safety flag + KB linking ----
      const tokensFor = (it: (typeof enriched)[number]) =>
        tokenize(normalizeTitle(it.title) + " " + (it.readableText || it.summaryExtractive));
      const simhashes = enriched.map((it) => simHash(tokensFor(it)));

      // Semantic vector: prefer dense embeddings when a deployment-level BYO key
      // (OPENAI_API_KEY) is configured; otherwise fall back to the local
      // hashing-trick vector. The pipeline never hard-depends on a key.
      const embedKey = process.env.OPENAI_API_KEY;
      let vectors: number[][] | null = null;
      if (embedKey && enriched.length) {
        const embedText = (it: (typeof enriched)[number]) =>
          `${it.title}. ${(it.readableText || it.summaryExtractive || "").slice(0, 1500)}`;
        vectors = await time("embed", enriched.length, () =>
          embedTexts({ provider: "openai", key: embedKey, texts: enriched.map(embedText), dimensions: 256 }),
        );
      }
      if (!vectors) vectors = enriched.map((it) => hashingVector(tokensFor(it)));

      // Wikidata linking only for representatives (cap API calls), shared cache.
      const kbCache = new Map<string, string | null>();
      const entityLinks = await time("kb-link", enriched.length, async () => {
        const out: { name: string; qid: string }[][] = enriched.map(() => []);
        for (let i = 0; i < enriched.length; i++) {
          const isRep = dedup.clusters[dedup.itemCluster[i]]?.representativeIndex === i;
          if (!isRep || enriched[i].entities.length === 0) continue;
          out[i] = await linkEntities(enriched[i].entities, kbCache, 3);
        }
        return out;
      });

      const itemPayloads = enriched.map((it, i) => ({
        dedupeKey: it.dedupeKey,
        sourceId: it.sourceId,
        sourceName: it.sourceName,
        kind: it.kind,
        title: it.title,
        url: it.url,
        canonicalUrl: it.canonicalUrl,
        summaryExtractive: it.summaryExtractive,
        readableText: it.readableText || undefined,
        // store the original source HTML alongside the cleaned text (capped to
        // bound storage); enables re-extraction / richer NER later.
        rawHtml: it.contentHtml ? it.contentHtml.slice(0, 16000) : undefined,
        contentHash: it.contentHash,
        image: it.image,
        author: it.author,
        lang: it.lang,
        publishedAt: it.publishedAt,
        wordCount: it.wordCount,
        topics: it.topics,
        entities: it.entities,
        entityLinks: entityLinks[i].length ? entityLinks[i] : undefined,
        contentType: it.contentType,
        engagement: { points: it.points ?? 0, comments: it.comments ?? 0 },
        features: features[i],
        simhash: simhashes[i],
        vector: vectors[i],
        flagged: isFlagged(it.title, it.readableText || it.summaryExtractive),
        clusterIndex: dedup.itemCluster[i],
        isRepresentative: dedup.clusters[dedup.itemCluster[i]]?.representativeIndex === i,
      }));

      // ---- Stage 6: persist ----
      const result = await time("persist", itemPayloads.length, () =>
        ctx.runMutation(internal.pipelineStore.storeResults, {
          fetchedAt: Date.now(),
          clusters: clusterPayloads,
          items: itemPayloads,
        }),
      );

      // housekeeping + learning + safety/observability ops
      await ctx.runMutation(internal.pipelineStore.pruneOld, { olderThanMs: RECENT_WINDOW_MS });
      await ctx.runMutation(internal.learning.recomputeSourceStats, {});
      await ctx.runMutation(internal.mlops.recomputeExperimentMetrics, {});
      await ctx.runMutation(internal.mlops.dataQualityCheck, {});
      await ctx.runMutation(internal.mlops.driftCheck, {});
      await ctx.runMutation(internal.mlops.autoDowngradeSources, {});

      await ctx.runMutation(internal.pipelineStore.finishRun, {
        runId,
        status: "ok",
        stages,
        counts: {
          fetched: raws.length,
          inserted: result.inserted,
          duplicates: result.duplicates,
          clusters: result.clusters,
        },
      });
      return result;
    } catch (e) {
      await ctx.runMutation(internal.pipelineStore.finishRun, {
        runId,
        status: "error",
        stages,
        counts: { fetched: 0, inserted: 0, duplicates: 0, clusters: 0 },
        error: (e as Error).message,
      });
      throw e;
    }
  },
});

/** Admin trigger for the dashboard "Run pipeline now" button. */
export const triggerRun = action({
  args: {},
  handler: async (ctx): Promise<{ inserted: number; duplicates: number; clusters: number }> => {
    // Actions have no db; authorize via an internal query (auth propagates).
    if (!(await ctx.runQuery(internal.authz.amIAdminInternal, {}))) {
      throw new Error("Admin access required");
    }
    return await ctx.runAction(internal.pipeline.runPipeline, { trigger: "manual" });
  },
});
