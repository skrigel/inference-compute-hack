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
import type { Chip, Facets, FreshDocument, HistogramBin, QueryEvent, RefineEvent } from "../lib/types";

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

export function useDashboard(seedQuery: string) {
  const cacheRef = useRef(new ScoreCache());
  const abortRef = useRef<AbortController | null>(null);
  const refineAbortRef = useRef<AbortController | null>(null);
  const chipSnapshotsRef = useRef(new Map<string, CachedScore[]>());
  const thresholdRef = useRef(0.5);
  const seededRef = useRef(false);

  const [predicate, setPredicate] = useState(seedQuery);
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
      setChips([]);
      setHasRun(true);
      setStreaming(true);
      setScanned(0);
      setEtaMs(0);
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
          setView(viewFromCache(cacheRef.current, thresholdRef.current));
          return;
        }
        if (event.type === "done") {
          const ms = event.elapsed_ms || Math.round(performance.now() - startedAt);
          setElapsedMs(ms);
          setScanned(event.scanned);
          setDocsPerSec(ms ? Math.round((event.scanned / ms) * 1000) : 0);
          setView(viewFromCache(cacheRef.current, thresholdRef.current));
          setLatencyMs(ms);
          setLatencyKind("cold");
          pushLatency(ms);
        }
      };

      try {
        await api.query(
          { predicate: nextPredicate, threshold: thresholdRef.current },
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
      await runRefineRequest({ utterance: trimmed });
    },
    [runRefineRequest],
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

  useEffect(() => {
    // Guard against React StrictMode's double mount-invoke firing the seed query twice.
    if (seededRef.current) return;
    seededRef.current = true;
    void runQuery(seedQuery);
    // run once on mount so the demo opens "live"
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
    mode: api.mode,
  };
}
