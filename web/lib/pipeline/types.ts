/**
 * Pipeline data contracts. These types are intentionally decoupled from the
 * Convex schema so pipeline stages stay pure and reusable: app/schema changes
 * don't ripple into the pipeline (a hard requirement of the challenge).
 *
 * Flow:  RawItem → NormalizedItem → EnrichedItem → (dedup) Cluster[] → ranked
 */

export type SourceKind = "rss" | "hackernews" | "reddit" | "x" | "newsletter" | "jsonfeed";

export interface SourceSpec {
  sourceId: string;
  name: string;
  kind: SourceKind;
  url: string;
  topics: string[];
  weight: number; // 0..1 reputation
  etag?: string;
  lastModified?: string;
}

/** What an ingest adapter returns per article, before normalization. */
export interface RawItem {
  sourceId: string;
  sourceName: string;
  kind: SourceKind;
  sourceWeight: number;
  title: string;
  url: string;
  contentHtml?: string;
  contentText?: string;
  author?: string;
  image?: string;
  publishedAt?: number; // ms epoch
  points?: number; // HN points / Reddit ups / X likes
  comments?: number;
  sourceTopics: string[];
}

export interface FetchResult {
  items: RawItem[];
  etag?: string;
  lastModified?: string;
  notModified?: boolean;
  error?: string;
}

export interface NormalizedItem extends RawItem {
  canonicalUrl: string;
  dedupeKey: string;
  summaryExtractive: string;
  readableText: string; // readability-extracted main content
  contentHash: string; // hash of title+content for update-diff trendlets
  lang: string;
  publishedAt: number;
  wordCount: number;
}

export interface EnrichedItem extends NormalizedItem {
  topics: string[];
  entities: string[];
  contentType: "news" | "opinion" | "release" | "discussion";
}

export interface ScoredItem extends EnrichedItem {
  features: {
    recency: number;
    sourceWeight: number;
    popularity: number;
    velocity: number;
  };
}
