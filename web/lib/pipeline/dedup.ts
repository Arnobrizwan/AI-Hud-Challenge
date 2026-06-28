import type { EnrichedItem } from "./types";
import { hashString, normalizeTitle, tokenize } from "./text";

/**
 * Stage 4 — deduplication & event grouping.
 * Stage A: MinHash signatures over title+lead word-shingles, indexed with LSH
 *          banding to find candidate near-duplicate pairs cheaply.
 * Stage B: confirm candidates by estimated Jaccard >= threshold.
 * Then union-find to form event clusters; elect a canonical representative.
 *
 * Concept ported from the Python repo's deduplication-service (LSH/MinHash).
 */

const NUM_HASHES = 48;
const BANDS = 12; // rows per band = NUM_HASHES / BANDS = 4
const SHINGLE_K = 3;
const JACCARD_THRESHOLD = 0.5;

// deterministic per-permutation seeds
const SEEDS = Array.from({ length: NUM_HASHES }, (_, i) => (i * 2654435761) >>> 0);

function shingles(item: EnrichedItem): Set<string> {
  const lead = (item.summaryExtractive || "").slice(0, 200);
  const toks = tokenize(normalizeTitle(item.title) + " " + lead);
  const set = new Set<string>();
  if (toks.length < SHINGLE_K) {
    toks.forEach((t) => set.add(t));
    return set;
  }
  for (let i = 0; i <= toks.length - SHINGLE_K; i++) {
    set.add(toks.slice(i, i + SHINGLE_K).join(" "));
  }
  return set;
}

function minhash(sh: Set<string>): number[] {
  const sig = new Array(NUM_HASHES).fill(Infinity);
  for (const s of sh) {
    const base = parseInt(hashString(s).slice(0, 13), 16);
    for (let i = 0; i < NUM_HASHES; i++) {
      const h = (base ^ SEEDS[i]) >>> 0;
      if (h < sig[i]) sig[i] = h;
    }
  }
  return sig;
}

function estJaccard(a: number[], b: number[]): number {
  let same = 0;
  for (let i = 0; i < NUM_HASHES; i++) if (a[i] === b[i]) same++;
  return same / NUM_HASHES;
}

export interface ClusterGroup {
  memberIndexes: number[];
  representativeIndex: number;
}

export interface DedupResult {
  clusters: ClusterGroup[];
  /** clusterId per item index (index into clusters[]) */
  itemCluster: number[];
}

class UnionFind {
  parent: number[];
  constructor(n: number) {
    this.parent = Array.from({ length: n }, (_, i) => i);
  }
  find(x: number): number {
    while (this.parent[x] !== x) {
      this.parent[x] = this.parent[this.parent[x]];
      x = this.parent[x];
    }
    return x;
  }
  union(a: number, b: number) {
    const ra = this.find(a), rb = this.find(b);
    if (ra !== rb) this.parent[ra] = rb;
  }
}

/** Stage 4 — cluster a batch of enriched items into events. */
export function dedupCluster(items: EnrichedItem[]): DedupResult {
  const n = items.length;
  const sigs = items.map((it) => minhash(shingles(it)));
  const uf = new UnionFind(n);

  // LSH: bucket by band signature; candidates share at least one band.
  const rows = Math.floor(NUM_HASHES / BANDS);
  const buckets = new Map<string, number[]>();
  for (let i = 0; i < n; i++) {
    for (let b = 0; b < BANDS; b++) {
      const band = sigs[i].slice(b * rows, (b + 1) * rows).join(",");
      const key = b + "|" + band;
      const arr = buckets.get(key);
      if (arr) arr.push(i);
      else buckets.set(key, [i]);
    }
  }
  for (const arr of buckets.values()) {
    if (arr.length < 2) continue;
    for (let i = 0; i < arr.length; i++) {
      for (let j = i + 1; j < arr.length; j++) {
        if (estJaccard(sigs[arr[i]], sigs[arr[j]]) >= JACCARD_THRESHOLD) {
          uf.union(arr[i], arr[j]);
        }
      }
    }
  }

  // collect clusters
  const byRoot = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const r = uf.find(i);
    const arr = byRoot.get(r);
    if (arr) arr.push(i);
    else byRoot.set(r, [i]);
  }

  const clusters: ClusterGroup[] = [];
  const itemCluster = new Array(n).fill(-1);
  for (const members of byRoot.values()) {
    // representative: highest source weight, then earliest publish time.
    let rep = members[0];
    for (const m of members) {
      const a = items[m], b = items[rep];
      if (
        a.sourceWeight > b.sourceWeight ||
        (a.sourceWeight === b.sourceWeight && a.publishedAt < b.publishedAt)
      ) {
        rep = m;
      }
    }
    const idx = clusters.length;
    clusters.push({ memberIndexes: members, representativeIndex: rep });
    for (const m of members) itemCluster[m] = idx;
  }

  return { clusters, itemCluster };
}
