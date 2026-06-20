import {
  HIST_BINS,
  type AggregateEvent,
  type DoneEvent,
  type FacetBucket,
  type HistogramBin,
  type QueryEvent,
  type QueryRequest,
  type ResultEvent,
} from "./types";

const MOCK_RESULTS: ResultEvent[] = [
  {
    type: "result",
    chunk_id: "mock_retry_0001",
    score: 0.91,
    meta: {
      type: "code",
      title: "urllib3/connectionpool.py",
      category: "python",
      year: 2023,
      path: "src/urllib3/connectionpool.py",
      lang: "python",
      repo: "urllib3",
    },
    rank: 0,
    rationale: null,
  },
  {
    type: "result",
    chunk_id: "mock_retry_0002",
    score: 0.78,
    meta: {
      type: "paper",
      title: "Retry Policies for Distributed Systems",
      category: "cs.DC",
      year: 2024,
      path: null,
      lang: null,
      repo: null,
    },
    rank: 1,
    rationale: null,
  },
  {
    type: "result",
    chunk_id: "mock_retry_0003",
    score: 0.63,
    meta: {
      type: "code",
      title: "requests/adapters.py",
      category: "python",
      year: 2022,
      path: "src/requests/adapters.py",
      lang: "python",
      repo: "requests",
    },
    rank: 2,
    rationale: null,
  },
];

export async function* queryMock(request: QueryRequest): AsyncGenerator<QueryEvent> {
  const threshold = request.threshold;
  const results = MOCK_RESULTS.filter((result) => result.score >= threshold);

  for (const result of results) {
    yield result;
  }

  yield makeAggregate(results, threshold);

  yield {
    type: "done",
    scanned: 128,
    matched: results.length,
    elapsed_ms: 24,
    warm: false,
    summary: `128 scanned - ${results.length} matched`,
  } satisfies DoneEvent;
}

function makeAggregate(results: ResultEvent[], threshold: number): AggregateEvent {
  return {
    type: "aggregate",
    scanned: 128,
    matched: results.length,
    histogram: makeHistogram(MOCK_RESULTS),
    facets: {
      type: makeFacet("type", results),
      category: makeFacet("category", results),
      year: makeFacet("year", results),
    },
    threshold,
    eta_ms: 0,
  };
}

function makeHistogram(results: ResultEvent[]): HistogramBin[] {
  const bins = Array.from({ length: HIST_BINS }, (_, index) => {
    const lo = index / HIST_BINS;
    const hi = (index + 1) / HIST_BINS;
    return { lo, hi, count: 0 };
  });

  for (const result of results) {
    const index = Math.min(HIST_BINS - 1, Math.floor(result.score * HIST_BINS));
    bins[index].count += 1;
  }

  return bins;
}

function makeFacet(
  key: "type" | "category" | "year",
  relevantResults: ResultEvent[],
): FacetBucket[] {
  const totals = new Map<string, number>();
  const relevant = new Map<string, number>();

  for (const result of MOCK_RESULTS) {
    const bucketKey = String(result.meta[key] ?? "unknown");
    totals.set(bucketKey, (totals.get(bucketKey) ?? 0) + 1);
  }

  for (const result of relevantResults) {
    const bucketKey = String(result.meta[key] ?? "unknown");
    relevant.set(bucketKey, (relevant.get(bucketKey) ?? 0) + 1);
  }

  return [...totals.entries()].map(([bucketKey, total]) => ({
    key: bucketKey,
    relevant: relevant.get(bucketKey) ?? 0,
    total,
  }));
}
