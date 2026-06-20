import {
  HIST_BINS,
  type ChunkMeta,
  type Facets,
  type FacetBucket,
  type HistogramBin,
  type ResultEvent,
} from "./types";

// The client-side score cache. Every streamed `result` event is upserted here,
// so once a scan completes the cache holds every scored chunk. Re-thresholding
// (the on-histogram drag) is then a pure recompute over this cache — ZERO
// inference, ZERO network. The functions below are deliberately free of any
// fetch/adapter import so that invariant is structural, not a promise.
export interface CachedScore {
  chunk_id: string;
  score: number;
  meta: ChunkMeta;
}

export class ScoreCache {
  private byId = new Map<string, CachedScore>();

  upsert(event: ResultEvent): void {
    this.byId.set(event.chunk_id, {
      chunk_id: event.chunk_id,
      score: event.score,
      meta: event.meta,
    });
  }

  upsertCached(entry: CachedScore): void {
    this.byId.set(entry.chunk_id, entry);
  }

  updateScore(chunkId: string, score: number): void {
    const current = this.byId.get(chunkId);
    if (!current) return;
    this.byId.set(chunkId, { ...current, score });
  }

  remove(chunkId: string): void {
    this.byId.delete(chunkId);
  }

  replaceAll(entries: CachedScore[]): void {
    this.byId = new Map(entries.map((entry) => [entry.chunk_id, entry]));
  }

  clear(): void {
    this.byId.clear();
  }

  size(): number {
    return this.byId.size;
  }

  all(): CachedScore[] {
    return [...this.byId.values()];
  }
}

export function recutHistogram(scores: CachedScore[]): HistogramBin[] {
  const counts = new Array<number>(HIST_BINS).fill(0);
  for (const entry of scores) {
    const index = Math.min(HIST_BINS - 1, Math.floor(entry.score * HIST_BINS));
    counts[index] += 1;
  }
  return counts.map((count, index) => ({
    lo: index / HIST_BINS,
    hi: (index + 1) / HIST_BINS,
    count,
  }));
}

function bucketKey(meta: ChunkMeta, facet: keyof Facets): string {
  const value = meta[facet];
  return value === null || value === undefined ? "unknown" : String(value);
}

function recutFacetGroup(scores: CachedScore[], facet: keyof Facets, threshold: number): FacetBucket[] {
  const totals = new Map<string, number>();
  const relevant = new Map<string, number>();
  for (const entry of scores) {
    const key = bucketKey(entry.meta, facet);
    totals.set(key, (totals.get(key) ?? 0) + 1);
    if (entry.score >= threshold) {
      relevant.set(key, (relevant.get(key) ?? 0) + 1);
    }
  }
  return [...totals.entries()]
    .map(([key, total]) => ({ key, relevant: relevant.get(key) ?? 0, total }))
    .sort((a, b) => b.relevant - a.relevant || b.total - a.total);
}

export function recutFacets(scores: CachedScore[], threshold: number): Facets {
  return {
    type: recutFacetGroup(scores, "type", threshold),
    category: recutFacetGroup(scores, "category", threshold),
    year: recutFacetGroup(scores, "year", threshold),
  };
}

export function matchedCount(scores: CachedScore[], threshold: number): number {
  return scores.reduce((total, entry) => total + (entry.score >= threshold ? 1 : 0), 0);
}

export function rankedSlice(scores: CachedScore[], limit: number, threshold = 0): CachedScore[] {
  return scores
    .filter((entry) => entry.score >= threshold)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}
