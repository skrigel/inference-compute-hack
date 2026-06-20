import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "../lib/api";
import {
  ScoreCache,
  matchedCount,
  rankedSlice,
  recutFacets,
  recutHistogram,
  type CachedScore,
} from "../lib/scoreCache";
import { selectFromCache } from "../lib/computeLab";
import type {
  BeamCandidate,
  Chip,
  Facets,
  FreshDocument,
  HistogramBin,
  QueryEvent,
  RefineEvent,
  Selection,
} from "../lib/types";

export type LatencyKind = "cold" | "warm" | "cached";
export type Tab = "rel" | "foot" | "perf";

const EMPTY_FACETS: Facets = { type: [], category: [], year: [] };
const FEED_LIMIT = 200;

export interface DashboardView {
  histogram: HistogramBin[];
  facets: Facets;
  matched: number;
  results: CachedScore[];
}

function viewFromCache(cache: ScoreCache, threshold: number): DashboardView {
  const all = cache.all();
  return {
    histogram: recutHistogram(all),
    facets: recutFacets(all, threshold),
    matched: matchedCount(all, threshold),
    results: rankedSlice(all, FEED_LIMIT, threshold),
  };
}

function readFileText(file: File): Promise<string> {
  if ("text" in file && typeof file.text === "function") {
    return file.text();
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error(`Failed to read ${file.name}`));
    reader.readAsText(file);
  });
}

const EMPTY_VIEW: DashboardView = {
  histogram: recutHistogram([]),
  facets: EMPTY_FACETS,
  matched: 0,
  results: [],
};

export function useDashboard() {
  const cacheRef = useRef(new ScoreCache());
  const abortRef = useRef<AbortController | null>(null);
  const refineAbortRef = useRef<AbortController | null>(null);
  const chipSnapshotsRef = useRef(new Map<string, CachedScore[]>());
  const thresholdRef = useRef(0.5);
  const computeBudgetRef = useRef(1);

  const [predicate, setPredicate] = useState("");
  const [threshold, setThresholdState] = useState(0.5);
  const [hasRun, setHasRun] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [refining, setRefining] = useState(false);
  const [scanned, setScanned] = useState(0);
  const [etaMs, setEtaMs] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [docsPerSec, setDocsPerSec] = useState(0);
  const [latencyMs, setLatencyMs] = useState(0);
  const [latencyKind, setLatencyKind] = useState<LatencyKind>("cold");
  const [latHistory, setLatHistory] = useState<number[]>([]);
  const [view, setView] = useState<DashboardView>(EMPTY_VIEW);
  const [chips, setChips] = useState<Chip[]>([]);
  const [activeTab, setActiveTab] = useState<Tab>("rel");
  // Axis 1 (Memory): compute budget + the resulting corpus scope.
  const [computeBudget, setComputeBudgetState] = useState(1);
  const [corpusScope, setCorpusScope] = useState<{ total: number; scored: number }>({ total: 0, scored: 0 });
  // Axis 3 (Truth): beam width over predicate combinations + explored candidates.
  const [beamWidth, setBeamWidth] = useState(1);
  const [beamCandidates, setBeamCandidates] = useState<BeamCandidate[] | null>(null);
  // Axis 2 (Movement): selection controls + the resulting selected set.
  const [precisionTarget, setPrecisionTarget] = useState(0.85);
  const [movementBudget, setMovementBudget] = useState(5);
  const [selectionBeamWidth, setSelectionBeamWidth] = useState(4);
  const [selection, setSelection] = useState<Selection | null>(null);

  const pushLatency = useCallback((ms: number) => {
    setLatHistory((prev) => [...prev, ms].slice(-16));
  }, []);

  const runQuery = useCallback(
    async (nextPredicate: string) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      cacheRef.current.clear();
      chipSnapshotsRef.current.clear();
      // Note: chips intentionally NOT cleared - refinements persist across queries
      setHasRun(true);
      setStreaming(true);
      setScanned(0);
      setEtaMs(0);
      setSelection(null);
      setBeamCandidates(null);
      setView(EMPTY_VIEW);

      const startedAt = performance.now();

      const onEvent = (event: QueryEvent) => {
        // Ignore events from a superseded/aborted stream (live or mock fallback)
        // so they can't upsert into the cleared cache or update state.
        if (controller.signal.aborted) return;
        if (event.type === "result") {
          cacheRef.current.upsert(event);
          return;
        }
        if (event.type === "aggregate") {
          setScanned(event.scanned);
          setEtaMs(event.eta_ms);
          if (event.corpus_total !== undefined && event.corpus_scored !== undefined) {
            setCorpusScope({ total: event.corpus_total, scored: event.corpus_scored });
          }
          setView(viewFromCache(cacheRef.current, thresholdRef.current));
          return;
        }
        if (event.type === "done") {
          const ms = event.elapsed_ms || Math.round(performance.now() - startedAt);
          setElapsedMs(ms);
          setScanned(event.scanned);
          if (event.corpus_total !== undefined && event.corpus_scored !== undefined) {
            setCorpusScope({ total: event.corpus_total, scored: event.corpus_scored });
          }
          setDocsPerSec(ms ? Math.round((event.scanned / ms) * 1000) : 0);
          setView(viewFromCache(cacheRef.current, thresholdRef.current));
          setLatencyMs(ms);
          setLatencyKind("cold");
          pushLatency(ms);
        }
      };

      try {
        await api.query(
          {
            predicate: nextPredicate,
            threshold: thresholdRef.current,
            compute_budget: computeBudgetRef.current,
          },
          onEvent,
          controller.signal,
        );
      } finally {
        if (abortRef.current === controller) setStreaming(false);
      }
    },
    [pushLatency],
  );

  // The load-bearing zero-inference path. Dragging the threshold ONLY recuts the
  // cache — it never touches `api`/fetch. scoreCache.test.ts guards that.
  const setThreshold = useCallback(
    (next: number) => {
      const clamped = Math.max(0, Math.min(1, next));
      thresholdRef.current = clamped;
      setThresholdState(clamped);
      setView(viewFromCache(cacheRef.current, clamped));
      setLatencyMs(5);
      setLatencyKind("cached");
      pushLatency(5);
    },
    [pushLatency],
  );

  const applyRefineEvent = useCallback(
    (event: RefineEvent, snapshot: CachedScore[]) => {
      if (event.type === "beam") {
        setBeamCandidates(event.candidates);
        return;
      }
      if (event.type === "chip") {
        chipSnapshotsRef.current.set(event.chip.clause_id, snapshot);
        setChips((prev) => [...prev, event.chip]);
        setLatencyMs(event.refine_ms);
        setLatencyKind(event.latency_kind);
        return;
      }
      if (event.type === "diff") {
        for (const chunkId of event.removed) cacheRef.current.remove(chunkId);
        for (const item of event.rescored) cacheRef.current.updateScore(item.chunk_id, item.score);
        for (const item of event.added) cacheRef.current.upsert(item);
        setView(viewFromCache(cacheRef.current, thresholdRef.current));
        return;
      }
      if (event.type === "aggregate") {
        setScanned(event.scanned);
        setEtaMs(event.eta_ms);
        setView(viewFromCache(cacheRef.current, thresholdRef.current));
        return;
      }
      if (event.type === "done") {
        setElapsedMs(event.elapsed_ms);
        setDocsPerSec(event.elapsed_ms ? Math.round((event.scanned / event.elapsed_ms) * 1000) : 0);
        setLatencyMs(event.elapsed_ms);
        setLatencyKind(event.warm ? "warm" : "cold");
        pushLatency(event.elapsed_ms);
      }
    },
    [pushLatency],
  );

  const runRefineRequest = useCallback(
    async (request: Parameters<typeof api.refine>[0]) => {
      refineAbortRef.current?.abort();
      const controller = new AbortController();
      refineAbortRef.current = controller;
      const snapshot = cacheRef.current.all();
      setBeamCandidates(null);
      setRefining(true);
      try {
        await api.refine(request, (event) => {
          if (controller.signal.aborted) return;
          applyRefineEvent(event, snapshot);
        }, controller.signal);
      } finally {
        if (refineAbortRef.current === controller) setRefining(false);
      }
    },
    [applyRefineEvent],
  );

  const runRefine = useCallback(
    async (utterance: string) => {
      const trimmed = utterance.trim();
      if (!trimmed) return;
      await runRefineRequest({ utterance: trimmed, beam_width: beamWidth });
    },
    [runRefineRequest, beamWidth],
  );

  const runClickRefine = useCallback(
    async (chunkId: string, sign: "+" | "-") => {
      await runRefineRequest({ click: { chunk_id: chunkId, sign } });
    },
    [runRefineRequest],
  );

  const removeChip = useCallback(
    async (clauseId: string) => {
      const response = await api.deleteClause(clauseId);
      if (!response.removed) return;
      const snapshot = chipSnapshotsRef.current.get(clauseId);
      if (snapshot) {
        cacheRef.current.replaceAll(snapshot);
        setView(viewFromCache(cacheRef.current, thresholdRef.current));
      }
      setChips((prev) => {
        const index = prev.findIndex((chip) => chip.clause_id === clauseId);
        if (index === -1) return prev;
        for (const chip of prev.slice(index)) chipSnapshotsRef.current.delete(chip.clause_id);
        return prev.slice(0, index);
      });
      setLatencyMs(response.refine_ms);
      setLatencyKind("cached");
      pushLatency(response.refine_ms);
    },
    [pushLatency],
  );

  const ingestFreshFiles = useCallback(
    async (files: File[] | FileList) => {
      const documents: FreshDocument[] = await Promise.all(
        Array.from(files).map(async (file) => ({
          title: file.name,
          text: await readFileText(file),
          type: "code",
          category: file.name.split(".").pop() || "fresh",
          year: new Date().getFullYear(),
          path: file.name,
          lang: file.name.split(".").pop() || null,
          repo: "fresh",
        })),
      );
      if (!documents.length) return;
      await api.ingest("demo", documents);
      await runQuery(predicate);
    },
    [predicate, runQuery],
  );

  const ingestCorpus = useCallback(
    async (corpusId: "demo" | "browsecomp", limit?: number) => {
      cacheRef.current.clear();
      chipSnapshotsRef.current.clear();
      setChips([]);
      setView(EMPTY_VIEW);
      setHasRun(false);
      await api.ingest(corpusId, [], limit);
    },
    [],
  );

  // Axis 1 (Memory): set the compute budget (corpus fraction scored per query).
  const setComputeBudget = useCallback((next: number) => {
    const clamped = Math.max(0.05, Math.min(1, next));
    computeBudgetRef.current = clamped;
    setComputeBudgetState(clamped);
  }, []);

  // Re-run the current query (used after committing a new compute budget).
  const rescan = useCallback(() => {
    void runQuery(predicate);
  }, [runQuery, predicate]);

  // Axis 2 (Movement) — Mode A: auto-threshold to the precision target. Pure
  // recut over the client cache (zero inference), then mark the selected set.
  const autoThreshold = useCallback(() => {
    const result = selectFromCache(cacheRef.current.all(), {
      mode: "threshold",
      precisionTarget,
      movementBudget,
      beamWidth: selectionBeamWidth,
    });
    setThreshold(result.threshold);
    setSelection(result);
  }, [precisionTarget, movementBudget, selectionBeamWidth, setThreshold]);

  // Axis 2 (Movement) — Mode B: max-coverage beam selection over the survivors.
  const smartSelect = useCallback(() => {
    const result = selectFromCache(cacheRef.current.all(), {
      mode: "smart",
      precisionTarget,
      movementBudget,
      beamWidth: selectionBeamWidth,
    });
    // Drop the slider to the survivor-pool boundary so the picked rows (which can
    // be lower-scoring, chosen for facet coverage) are actually visible and the
    // highlight lands inside the same pool the beam searched over.
    setThreshold(result.threshold);
    setSelection(result);
  }, [precisionTarget, movementBudget, selectionBeamWidth, setThreshold]);

  const clearSelection = useCallback(() => setSelection(null), []);

  return {
    predicate,
    setPredicate,
    threshold,
    setThreshold,
    hasRun,
    streaming,
    refining,
    scanned,
    etaMs,
    elapsedMs,
    docsPerSec,
    latencyMs,
    latencyKind,
    latHistory,
    view,
    chips,
    activeTab,
    setActiveTab,
    runQuery,
    runRefine,
    runClickRefine,
    removeChip,
    ingestFreshFiles,
    ingestCorpus,
    // Axis 1 (Memory)
    computeBudget,
    setComputeBudget,
    corpusScope,
    rescan,
    // Axis 3 (Truth)
    beamWidth,
    setBeamWidth,
    beamCandidates,
    // Axis 2 (Movement)
    precisionTarget,
    setPrecisionTarget,
    movementBudget,
    setMovementBudget,
    selectionBeamWidth,
    setSelectionBeamWidth,
    selection,
    autoThreshold,
    smartSelect,
    clearSelection,
    mode: api.mode,
  };
}
