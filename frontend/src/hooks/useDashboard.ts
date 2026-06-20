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
import type { Facets, HistogramBin, QueryEvent } from "../lib/types";

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

const EMPTY_VIEW: DashboardView = {
  histogram: recutHistogram([]),
  facets: EMPTY_FACETS,
  matched: 0,
  results: [],
};

export function useDashboard(seedQuery: string) {
  const cacheRef = useRef(new ScoreCache());
  const abortRef = useRef<AbortController | null>(null);
  const thresholdRef = useRef(0.5);
  const seededRef = useRef(false);

  const [predicate, setPredicate] = useState(seedQuery);
  const [threshold, setThresholdState] = useState(0.5);
  const [hasRun, setHasRun] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [scanned, setScanned] = useState(0);
  const [etaMs, setEtaMs] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [docsPerSec, setDocsPerSec] = useState(0);
  const [latencyMs, setLatencyMs] = useState(0);
  const [latencyKind, setLatencyKind] = useState<LatencyKind>("cold");
  const [latHistory, setLatHistory] = useState<number[]>([]);
  const [view, setView] = useState<DashboardView>(EMPTY_VIEW);
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
    scanned,
    etaMs,
    elapsedMs,
    docsPerSec,
    latencyMs,
    latencyKind,
    latHistory,
    view,
    activeTab,
    setActiveTab,
    runQuery,
    mode: api.mode,
  };
}
