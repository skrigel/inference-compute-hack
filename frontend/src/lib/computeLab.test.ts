import { describe, expect, it } from "vitest";
import type { ChunkMeta } from "./types";
import type { CachedScore } from "./scoreCache";
import { autoThreshold, maxCoverageSelect, selectFromCache } from "./computeLab";

function meta(over: Partial<ChunkMeta>): ChunkMeta {
  return {
    type: "code",
    title: "t",
    category: null,
    year: null,
    path: null,
    lang: null,
    repo: null,
    ...over,
  };
}

function entry(id: string, score: number, over: Partial<ChunkMeta> = {}): CachedScore {
  return { chunk_id: id, score, meta: meta(over) };
}

describe("autoThreshold", () => {
  it("keeps the largest best-first prefix whose mean stays >= the target", () => {
    const { threshold, count } = autoThreshold([0.9, 0.8, 0.4, 0.2], 0.7);
    // 0.9 -> mean 0.90, +0.8 -> 0.85, +0.4 -> 0.70 (still >=), +0.2 -> 0.575 (stop)
    expect(count).toBe(3);
    expect(threshold).toBeCloseTo(0.4, 6);
  });

  it("returns count 0 when nothing meets the target", () => {
    const { count } = autoThreshold([0.1, 0.05], 0.9);
    expect(count).toBe(0);
  });

  it("handles an empty score list", () => {
    expect(autoThreshold([], 0.8)).toEqual({ threshold: 1, count: 0 });
  });
});

describe("maxCoverageSelect", () => {
  it("never scores worse than the greedy floor", () => {
    const items = [
      { id: "a", tokens: new Set(["type:code", "lang:python"]), score: 0.9 },
      { id: "b", tokens: new Set(["type:code", "lang:python"]), score: 0.8 },
      { id: "c", tokens: new Set(["type:paper", "category:rl"]), score: 0.5 },
    ];
    const result = maxCoverageSelect(items, 2, 4);
    expect(result.objective).toBeGreaterThanOrEqual(result.greedyObjective);
    // Best 2-subset for coverage favors a + c (4 distinct facets).
    expect(result.covered.length).toBe(4);
    expect(result.selected.length).toBe(2);
  });

  it("returns empty selection when the movement budget is zero", () => {
    const items = [{ id: "a", tokens: new Set(["type:code"]), score: 1 }];
    expect(maxCoverageSelect(items, 0, 4)).toEqual({
      selected: [],
      covered: [],
      objective: 0,
      greedyObjective: 0,
    });
  });
});

describe("selectFromCache", () => {
  const entries = [
    entry("a", 0.95, { type: "code", lang: "python" }),
    entry("b", 0.9, { type: "code", lang: "python" }),
    entry("c", 0.6, { type: "paper", category: "rl" }),
    entry("d", 0.1, { type: "code", lang: "go" }),
  ];

  it("threshold mode returns the best-first prefix above the auto threshold", () => {
    const selection = selectFromCache(entries, {
      mode: "threshold",
      precisionTarget: 0.85,
      movementBudget: 5,
      beamWidth: 4,
    });
    expect(selection.mode).toBe("threshold");
    expect(selection.selectedIds).toEqual(["a", "b"]);
    expect(selection.coveredFacets).toEqual([]);
  });

  it("smart mode maximizes facet coverage within the movement budget", () => {
    const selection = selectFromCache(entries, {
      mode: "smart",
      precisionTarget: 0.5,
      movementBudget: 2,
      beamWidth: 4,
    });
    expect(selection.mode).toBe("smart");
    expect(selection.selectedIds.length).toBeLessThanOrEqual(2);
    expect(selection.objective).toBeGreaterThanOrEqual(selection.greedyObjective);
    expect(selection.coveredFacets.length).toBeGreaterThan(0);
  });
});
