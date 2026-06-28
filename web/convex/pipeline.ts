import { internalAction, action } from "./_generated/server";
import { v } from "convex/values";
import { internal } from "./_generated/api";
import { getAuthUserId } from "@convex-dev/auth/server";

import { ingestSource } from "../lib/pipeline/ingest";
import { normalizeBatch } from "../lib/pipeline/normalize";
import { enrichBatch } from "../lib/pipeline/enrich";
import { dedupCluster } from "../lib/pipeline/dedup";
import { computeItemFeatures } from "../lib/pipeline/rank";
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
      const raws: RawItem[] = await time("ingest", sources.length, async () => {
        const all: RawItem[] = [];
        for (const s of sources) {
          // be polite: stagger requests, extra spacing for rate-limited hosts
          await sleep(s.kind === "reddit" || s.kind === "x" ? 700 : 250);
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
          if (res.items.length) all.push(...res.items);
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

      const itemPayloads = enriched.map((it, i) => ({
        dedupeKey: it.dedupeKey,
        sourceId: it.sourceId,
        sourceName: it.sourceName,
        kind: it.kind,
        title: it.title,
        url: it.url,
        canonicalUrl: it.canonicalUrl,
        summaryExtractive: it.summaryExtractive,
        image: it.image,
        author: it.author,
        lang: it.lang,
        publishedAt: it.publishedAt,
        wordCount: it.wordCount,
        topics: it.topics,
        entities: it.entities,
        contentType: it.contentType,
        engagement: { points: it.points ?? 0, comments: it.comments ?? 0 },
        features: features[i],
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

      // housekeeping
      await ctx.runMutation(internal.pipelineStore.pruneOld, { olderThanMs: RECENT_WINDOW_MS });

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

/** Public trigger for the dashboard "Run pipeline now" button. */
export const triggerRun = action({
  args: {},
  handler: async (ctx): Promise<{ inserted: number; duplicates: number; clusters: number }> => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Not authenticated");
    return await ctx.runAction(internal.pipeline.runPipeline, { trigger: "manual" });
  },
});
