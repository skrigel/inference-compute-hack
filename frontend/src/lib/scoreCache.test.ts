import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ScoreCache,
  matchedCount,
  rankedSlice,
  recutFacets,
  recutHistogram,
  type CachedScore,
} from "./scoreCache";
import { HIST_BINS, type ResultEvent } from "./types";

function result(chunk_id: string, score: number, type: "paper" | "code", category: string): ResultEvent {
  return {
    type: "result",
    chunk_id,
    score,
    meta: { type, title: chunk_id, category, year: 2024, path: null, lang: null, repo: null },
    rank: 0,
    rationale: null,
  };
}

function sampleCache(): ScoreCache {
  const cache = new ScoreCache();
  cache.upsert(result("a", 0.92, "code", "python"));
  cache.upsert(result("b", 0.61, "paper", "cs.IR"));
  cache.upsert(result("c", 0.33, "code", "python"));
  cache.upsert(result("d", 0.12, "paper", "cs.LG"));
  return cache;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("scoreCache recut", () => {
  it("histogram has HIST_BINS bins and counts every cached score", () => {
    const bins = recutHistogram(sampleCache().all());
    expect(bins).toHaveLength(HIST_BINS);
    expect(bins.reduce((sum, bin) => sum + bin.count, 0)).toBe(4);
  });

  it("matched count and facets shift with the threshold (no re-scoring)", () => {
    const all = sampleCache().all();
    expect(matchedCount(all, 0.5)).toBe(2);
    expect(matchedCount(all, 0.2)).toBe(3);

    const facets = recutFacets(all, 0.5);
    const code = facets.type.find((bucket) => bucket.key === "code");
    expect(code).toEqual({ key: "code", relevant: 1, total: 2 });
  });

  it("ranked slice is best-first and respects the limit", () => {
    const ranked = rankedSlice(sampleCache().all(), 2);
    expect(ranked.map((entry: CachedScore) => entry.chunk_id)).toEqual(["a", "b"]);
  });

  it("ranked slice filters out scores below the active threshold", () => {
    const ranked = rankedSlice(sampleCache().all(), 10, 0.5);
    expect(ranked.map((entry: CachedScore) => entry.chunk_id)).toEqual(["a", "b"]);
  });

  it("re-cutting the threshold performs ZERO network calls (the headline invariant)", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch" as never);
    const all = sampleCache().all();

    // Simulate dragging the threshold across the full range.
    for (let threshold = 0; threshold <= 1; threshold += 0.1) {
      recutHistogram(all);
      recutFacets(all, threshold);
      matchedCount(all, threshold);
      rankedSlice(all, 200);
    }

    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
