import {
  HIST_BINS,
  type AggregateEvent,
  type DoneEvent,
  type Facets,
  type FacetBucket,
  type HistogramBin,
  type QueryEvent,
  type QueryRequest,
  type ResultEvent,
} from "./types";

// Stagger between streamed results so the dashboard visibly fills on mock —
// ~24 results x 28ms ≈ a ~700ms "cold" scan feel, matching the demo fallback.
const STREAM_DELAY_MS = 28;

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

interface MockItem {
  chunk_id: string;
  score: number;
  type: "paper" | "code";
  title: string;
  category: string;
  year: number;
  path: string | null;
  repo: string | null;
}

// A spread of ~24 items: varied scores (0.1–0.97), both modalities, several
// categories and years, with the retry/backoff/networking items scored high so
// the seeded demo predicate lands convincingly.
const MOCK_ITEMS: MockItem[] = [
  { chunk_id: "m01", score: 0.96, type: "code", title: "urllib3/connectionpool.py", category: "python", year: 2023, path: "src/urllib3/connectionpool.py", repo: "urllib3" },
  { chunk_id: "m02", score: 0.93, type: "code", title: "requests/adapters.py", category: "python", year: 2022, path: "src/requests/adapters.py", repo: "requests" },
  { chunk_id: "m03", score: 0.9, type: "code", title: "tenacity/retry.py", category: "python", year: 2024, path: "src/tenacity/retry.py", repo: "tenacity" },
  { chunk_id: "m04", score: 0.86, type: "code", title: "aiohttp/client.py", category: "python", year: 2023, path: "aiohttp/client.py", repo: "aiohttp" },
  { chunk_id: "m05", score: 0.82, type: "paper", title: "Retry Policies for Distributed Systems", category: "cs.DC", year: 2024, path: null, repo: null },
  { chunk_id: "m06", score: 0.79, type: "code", title: "grpc/retry_interceptor.go", category: "go", year: 2023, path: "grpc/retry_interceptor.go", repo: "grpc-go" },
  { chunk_id: "m07", score: 0.74, type: "code", title: "httpx/_transports/default.py", category: "python", year: 2024, path: "httpx/_transports/default.py", repo: "httpx" },
  { chunk_id: "m08", score: 0.71, type: "paper", title: "Exponential Backoff and Jitter", category: "cs.NI", year: 2019, path: null, repo: null },
  { chunk_id: "m09", score: 0.66, type: "code", title: "kubernetes/client/retry.go", category: "go", year: 2022, path: "client-go/util/retry/util.go", repo: "client-go" },
  { chunk_id: "m10", score: 0.62, type: "paper", title: "Resilient Microservice Networking", category: "cs.DC", year: 2021, path: null, repo: null },
  { chunk_id: "m11", score: 0.58, type: "code", title: "boto3/retries/standard.py", category: "python", year: 2023, path: "boto3/retries/standard.py", repo: "boto3" },
  { chunk_id: "m12", score: 0.54, type: "paper", title: "Information Retrieval Ranking Metrics", category: "cs.IR", year: 2023, path: null, repo: null },
  { chunk_id: "m13", score: 0.49, type: "code", title: "frontend/Histogram.tsx", category: "typescript", year: 2024, path: "src/Histogram.tsx", repo: "demo" },
  { chunk_id: "m14", score: 0.45, type: "paper", title: "Dense Retrieval for Code Search", category: "cs.IR", year: 2022, path: null, repo: null },
  { chunk_id: "m15", score: 0.41, type: "code", title: "rust/reqwest/retry.rs", category: "rust", year: 2024, path: "reqwest/src/retry.rs", repo: "reqwest" },
  { chunk_id: "m16", score: 0.37, type: "paper", title: "Prefix Caching for Interactive Search", category: "cs.LG", year: 2024, path: null, repo: null },
  { chunk_id: "m17", score: 0.33, type: "code", title: "logging/handlers.py", category: "python", year: 2020, path: "lib/logging/handlers.py", repo: "cpython" },
  { chunk_id: "m18", score: 0.29, type: "paper", title: "Speculative Decoding Throughput", category: "cs.LG", year: 2023, path: null, repo: null },
  { chunk_id: "m19", score: 0.25, type: "code", title: "db/connection.py", category: "python", year: 2021, path: "db/connection.py", repo: "sqlalchemy" },
  { chunk_id: "m20", score: 0.21, type: "paper", title: "Quantization for Fast Inference", category: "cs.LG", year: 2024, path: null, repo: null },
  { chunk_id: "m21", score: 0.18, type: "code", title: "ui/theme.css", category: "css", year: 2024, path: "src/theme.css", repo: "demo" },
  { chunk_id: "m22", score: 0.14, type: "paper", title: "Attention and Long-Context Recall", category: "cs.CL", year: 2024, path: null, repo: null },
  { chunk_id: "m23", score: 0.11, type: "code", title: "math/statistics.py", category: "python", year: 2019, path: "lib/statistics.py", repo: "cpython" },
  { chunk_id: "m24", score: 0.08, type: "paper", title: "Diffusion Language Models", category: "cs.CL", year: 2025, path: null, repo: null },
];

function toResult(item: MockItem, rank: number): ResultEvent {
  return {
    type: "result",
    chunk_id: item.chunk_id,
    score: item.score,
    meta: {
      type: item.type,
      title: item.title,
      category: item.category,
      year: item.year,
      path: item.path,
      lang: item.type === "code" ? item.category : null,
      repo: item.repo,
    },
    rank,
    rationale: null,
  };
}

const MOCK_RESULTS: ResultEvent[] = [...MOCK_ITEMS]
  .sort((a, b) => b.score - a.score)
  .map((item, index) => toResult(item, index));

export async function ingestMock(_corpusId: string): Promise<{ n_chunks: number; facets: Facets }> {
  return { n_chunks: MOCK_RESULTS.length, facets: allFacets() };
}

function allFacets(): Facets {
  return {
    type: makeFacet("type", MOCK_RESULTS),
    category: makeFacet("category", MOCK_RESULTS),
    year: makeFacet("year", MOCK_RESULTS),
  };
}

export async function* queryMock(
  request: QueryRequest,
  signal?: AbortSignal,
): AsyncGenerator<QueryEvent> {
  const startedAt = Date.now();
  // Stream EVERY scored chunk best-first (like the backend) so the client cache
  // is complete and threshold drag is a pure recut.
  for (const result of MOCK_RESULTS) {
    if (signal?.aborted) return;
    yield result;
    await sleep(STREAM_DELAY_MS);
  }

  const matched = MOCK_RESULTS.filter((result) => result.score >= request.threshold);
  yield makeAggregate(matched, request.threshold);

  yield {
    type: "done",
    scanned: MOCK_RESULTS.length,
    matched: matched.length,
    elapsed_ms: Date.now() - startedAt,
    warm: false,
    summary: `${MOCK_RESULTS.length} scanned · ${matched.length} matched`,
  } satisfies DoneEvent;
}

function makeAggregate(matched: ResultEvent[], threshold: number): AggregateEvent {
  return {
    type: "aggregate",
    scanned: MOCK_RESULTS.length,
    matched: matched.length,
    histogram: makeHistogram(MOCK_RESULTS),
    facets: {
      type: makeFacet("type", matched),
      category: makeFacet("category", matched),
      year: makeFacet("year", matched),
    },
    threshold,
    eta_ms: 0,
  };
}

function makeHistogram(results: ResultEvent[]): HistogramBin[] {
  const bins = Array.from({ length: HIST_BINS }, (_, index) => ({
    lo: index / HIST_BINS,
    hi: (index + 1) / HIST_BINS,
    count: 0,
  }));
  for (const result of results) {
    const index = Math.min(HIST_BINS - 1, Math.floor(result.score * HIST_BINS));
    bins[index].count += 1;
  }
  return bins;
}

function makeFacet(key: "type" | "category" | "year", relevantResults: ResultEvent[]): FacetBucket[] {
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
  return [...totals.entries()]
    .map(([bucketKey, total]) => ({ key: bucketKey, relevant: relevant.get(bucketKey) ?? 0, total }))
    .sort((a, b) => b.relevant - a.relevant || b.total - a.total);
}
