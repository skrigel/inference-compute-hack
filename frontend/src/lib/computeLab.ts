// Axis 2 (Movement): zero-inference selection over the complete client score
// cache. Mirrors backend/select.py exactly so mock and live behave identically.
// Pure arithmetic — no fetch/adapter imports — keeping the "score first, then
// decide what to move" inversion structural.
import type { ChunkMeta, Selection, SelectMode } from "./types";
import type { CachedScore } from "./scoreCache";

export interface AutoThreshold {
  threshold: number;
  count: number;
}

// Mode A: the largest best-first prefix whose mean score stays >= the precision
// target. The returned threshold is the lowest score included.
export function autoThreshold(scores: number[], precisionTarget: number): AutoThreshold {
  const ordered = [...scores].sort((a, b) => b - a);
  if (ordered.length === 0) return { threshold: 1, count: 0 };
  let running = 0;
  let bestK = 0;
  for (let k = 1; k <= ordered.length; k++) {
    running += ordered[k - 1];
    if (running / k >= precisionTarget) {
      bestK = k;
    } else {
      break;
    }
  }
  if (bestK === 0) return { threshold: Math.min(1, ordered[0] + 1e-9), count: 0 };
  return { threshold: ordered[bestK - 1], count: bestK };
}

export function facetTokens(meta: ChunkMeta): Set<string> {
  const tokens = new Set<string>([`type:${meta.type}`]);
  for (const name of ["category", "year", "repo", "lang"] as const) {
    const value = meta[name];
    if (value !== null && value !== undefined) tokens.add(`${name}:${value}`);
  }
  return tokens;
}

interface CoverageItem {
  id: string;
  tokens: Set<string>;
  score: number;
}

interface Beam {
  selected: string[];
  covered: Set<string>;
  scoreSum: number;
}

// Lexicographic objective: facet coverage first, total score as tie-break.
function better(a: Beam, b: Beam): boolean {
  if (a.covered.size !== b.covered.size) return a.covered.size > b.covered.size;
  return a.scoreSum > b.scoreSum;
}

function union(a: Set<string>, b: Set<string>): Set<string> {
  const out = new Set(a);
  for (const value of b) out.add(value);
  return out;
}

export interface CoverageResult {
  selected: string[];
  covered: string[];
  objective: number;
  greedyObjective: number;
}

// Mode B: beam search over subsets of size <= K maximising facet coverage.
export function maxCoverageSelect(
  items: CoverageItem[],
  movementBudget: number,
  beamWidth: number,
): CoverageResult {
  const budget = Math.max(0, movementBudget);
  const width = Math.max(1, beamWidth);
  if (items.length === 0 || budget === 0) {
    return { selected: [], covered: [], objective: 0, greedyObjective: 0 };
  }

  const search = (beamSize: number): Beam => {
    let beams: Beam[] = [{ selected: [], covered: new Set(), scoreSum: 0 }];
    let best: Beam = { selected: [], covered: new Set(), scoreSum: 0 };
    const rounds = Math.min(budget, items.length);
    for (let round = 0; round < rounds; round++) {
      const expanded = new Map<string, Beam>();
      for (const beam of beams) {
        const selectedSet = new Set(beam.selected);
        for (const item of items) {
          if (selectedSet.has(item.id)) continue;
          const next: Beam = {
            selected: [...beam.selected, item.id],
            covered: union(beam.covered, item.tokens),
            scoreSum: beam.scoreSum + item.score,
          };
          const key = [...next.selected].sort().join("|");
          const existing = expanded.get(key);
          if (!existing || better(next, existing)) expanded.set(key, next);
        }
      }
      if (expanded.size === 0) break;
      const ranked = [...expanded.values()].sort((a, b) => (better(a, b) ? -1 : 1));
      beams = ranked.slice(0, beamSize);
      if (better(beams[0], best)) best = beams[0];
    }
    return best;
  };

  const greedy = search(1);
  if (width === 1) {
    return {
      selected: greedy.selected,
      covered: [...greedy.covered].sort(),
      objective: greedy.scoreSum,
      greedyObjective: greedy.scoreSum,
    };
  }
  const beam = search(width);
  const winner = better(beam, greedy) ? beam : greedy;
  return {
    selected: winner.selected,
    covered: [...winner.covered].sort(),
    objective: winner.scoreSum,
    greedyObjective: greedy.scoreSum,
  };
}

export interface SelectParams {
  mode: SelectMode;
  precisionTarget: number;
  movementBudget: number;
  beamWidth: number;
}

// The single entry point useDashboard calls — operates on the complete cache.
export function selectFromCache(entries: CachedScore[], params: SelectParams): Selection {
  const { threshold } = autoThreshold(
    entries.map((entry) => entry.score),
    params.precisionTarget,
  );

  if (params.mode === "threshold") {
    const selectedIds = entries
      .filter((entry) => entry.score >= threshold)
      .sort((a, b) => b.score - a.score)
      .map((entry) => entry.chunk_id);
    return {
      mode: "threshold",
      threshold,
      selectedIds,
      coveredFacets: [],
      objective: 0,
      greedyObjective: 0,
      movementBudget: 0,
      beamWidth: 1,
      candidatePool: selectedIds.length,
    };
  }

  const pool: CoverageItem[] = entries
    .filter((entry) => entry.score >= threshold)
    .sort((a, b) => b.score - a.score)
    .map((entry) => ({ id: entry.chunk_id, tokens: facetTokens(entry.meta), score: entry.score }));
  const coverage = maxCoverageSelect(pool, params.movementBudget, params.beamWidth);
  return {
    mode: "smart",
    threshold,
    selectedIds: coverage.selected,
    coveredFacets: coverage.covered,
    objective: coverage.objective,
    greedyObjective: coverage.greedyObjective,
    movementBudget: params.movementBudget,
    beamWidth: params.beamWidth,
    candidatePool: pool.length,
  };
}
