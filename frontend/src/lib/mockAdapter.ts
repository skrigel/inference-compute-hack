import {
  HIST_BINS,
  type AggregateEvent,
  type BeamCandidate,
  type BeamEvent,
  type DoneEvent,
  type Facets,
  type FacetBucket,
  type FreshDocument,
  type HistogramBin,
  type ChipEvent,
  type DiffEvent,
  type QueryEvent,
  type QueryRequest,
  type RefineEvent,
  type RefineRequest,
  type ResultEvent,
} from "./types";
import { classifyRefine } from "./classify";
import { facetTokens } from "./computeLab";

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
const DEMO_ITEMS: MockItem[] = [
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

// BrowseComp+ mock items - simulated web documents
const BROWSECOMP_ITEMS: MockItem[] = Array.from({ length: 100 }, (_, i) => ({
  chunk_id: `bcp-${i}`,
  score: Math.max(0.05, 0.95 - (i * 0.009) + (Math.sin(i) * 0.1)),
  type: "paper" as const,
  title: `BrowseComp Document ${i + 1}`,
  category: ["news", "wiki", "blog", "academic", "forum"][i % 5],
  year: 2020 + (i % 6),
  path: `https://example.com/doc/${i}`,
  repo: null,
}));

let activeCorpus: "demo" | "browsecomp" = "demo";
let freshItems: MockItem[] = [];
let clauseSeq = 1;

function getBaseItems(): MockItem[] {
  return activeCorpus === "browsecomp" ? BROWSECOMP_ITEMS : DEMO_ITEMS;
}

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

function mockItems(): MockItem[] {
  return [...getBaseItems(), ...freshItems];
}

function mockResults(): ResultEvent[] {
  return [...mockItems()].sort((a, b) => b.score - a.score).map((item, index) => toResult(item, index));
}

export async function ingestMock(
  corpusId: string,
  documents: FreshDocument[] = [],
): Promise<{ n_chunks: number; facets: Facets }> {
  // Switch corpus based on corpusId
  if (corpusId === "browsecomp" || corpusId === "demo") {
    activeCorpus = corpusId;
    freshItems = [];
  }
  if (documents.length > 0) {
    freshItems = documents.map((document, index) => ({
      chunk_id: `fresh-${Date.now()}-${index}`,
      score: document.text.toLowerCase().includes("sentinel") ? 0.97 : 0.72,
      type: document.type,
      title: document.title,
      category: document.category ?? document.lang ?? "fresh",
      year: document.year ?? new Date().getFullYear(),
      path: document.path,
      repo: document.repo,
    }));
  }
  return { n_chunks: mockResults().length, facets: allFacets() };
}

function allFacets(): Facets {
  const results = mockResults();
  return {
    type: makeFacet("type", results),
    category: makeFacet("category", results),
    year: makeFacet("year", results),
  };
}

export async function* queryMock(
  request: QueryRequest,
  signal?: AbortSignal,
): AsyncGenerator<QueryEvent> {
  const startedAt = Date.now();
  const allItems = mockItems();
  // Axis 1 (Memory): score only the budgeted prefix of the corpus.
  const budget = Math.min(1, Math.max(0, request.compute_budget ?? 1));
  const inScope =
    budget >= 1 ? allItems.length : Math.min(allItems.length, Math.max(1, Math.ceil(budget * allItems.length)));
  const scopedItems = allItems.slice(0, inScope);
  const results = [...scopedItems]
    .sort((a, b) => b.score - a.score)
    .map((item, index) => toResult(item, index));
  const corpus = { total: allItems.length, scored: results.length, budget };
  // Stream EVERY scored chunk best-first (like the backend) so the client cache
  // is complete and threshold drag is a pure recut.
  for (const result of results) {
    if (signal?.aborted) return;
    yield result;
    await sleep(STREAM_DELAY_MS);
  }

  const matched = results.filter((result) => result.score >= request.threshold);
  yield makeAggregate(matched, request.threshold, results, corpus);

  yield {
    type: "done",
    scanned: results.length,
    matched: matched.length,
    elapsed_ms: Date.now() - startedAt,
    warm: false,
    summary: `${results.length} scanned · ${matched.length} matched`,
    corpus_total: corpus.total,
    corpus_scored: corpus.scored,
    compute_budget: corpus.budget,
  } satisfies DoneEvent;
}

export async function* refineMock(
  request: RefineRequest,
  signal?: AbortSignal,
): AsyncGenerator<RefineEvent> {
  const startedAt = Date.now();
  const results = mockResults();
  // Axis 3 (Truth): beam_width > 1 explores candidate clauses and the objective
  // function selects the winner before the usual chip/diff/aggregate/done.
  let effectiveRequest = request;
  let beamEvent: BeamEvent | null = null;
  if (request.utterance && (request.beam_width ?? 1) > 1) {
    const beam = mockBeam(request.utterance, request.beam_width as number, results);
    beamEvent = beam.event;
    effectiveRequest = { ...request, utterance: beam.winnerText };
  }
  const { operation, text, confidence } = refineIntent(effectiveRequest);
  const clauseId = `m-c${clauseSeq++}`;
  const chip = {
    clause_id: clauseId,
    op: operation,
    text,
    label: operation[0].toUpperCase() + operation.slice(1),
    removable: true,
    confidence,
  };
  const refineMs = request.click ? 4 : 180;
  const diff = makeRefineDiff(effectiveRequest, results);
  const nextResults = applyMockDiff(results, diff);
  const matched = nextResults.filter((result) => result.score >= 0.5);

  if (signal?.aborted) return;
  if (beamEvent) {
    yield beamEvent;
    await sleep(60);
    if (signal?.aborted) return;
  }
  yield {
    type: "chip",
    operation,
    chip,
    refine_ms: refineMs,
    latency_kind: request.click ? "cached" : "warm",
  } satisfies ChipEvent;
  await sleep(request.click ? 4 : 80);
  if (signal?.aborted) return;
  yield { ...diff, refine_ms: refineMs } satisfies DiffEvent;
  yield makeAggregate(matched, 0.5);
  yield {
    type: "done",
    scanned: diff.rescored.length,
    matched: matched.length,
    elapsed_ms: refineMs || Date.now() - startedAt,
    warm: true,
    summary: `${diff.rescored.length} rescored · ${matched.length} matched`,
  } satisfies DoneEvent;
}

// Axis 3 (Truth): a deterministic mock of the candidate-clause beam search.
// Candidate 0 is the raw utterance; the rest are facet-narrowed variants drawn
// from the survivors. Narrowing trades coverage for precision (objective).
function mockBeam(
  utterance: string,
  beamWidth: number,
  results: ResultEvent[],
): { event: BeamEvent; winnerText: string } {
  const survivors = results.filter((result) => result.score >= 0.5);
  const base = utterance.trim();
  const counts = new Map<string, number>();
  for (const result of survivors) {
    for (const token of facetTokens(result.meta)) counts.set(token, (counts.get(token) ?? 0) + 1);
  }
  const variants = [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([token]) => {
      const [name, value] = token.split(":");
      return `${base} (${name} ${value})`;
    });
  const texts = [base, ...variants].slice(0, Math.max(1, beamWidth));
  const round = (value: number) => Math.round(value * 1000) / 1000;
  const baseObjective = survivors.length
    ? survivors.reduce((sum, result) => sum + result.score, 0) / survivors.length
    : 0;
  const candidates: BeamCandidate[] = texts.map((candidateText, index) => {
    if (index === 0) {
      return { text: candidateText, objective: round(baseObjective), coverage: 1, selected: survivors.length, chosen: false };
    }
    const coverage = Math.max(0.2, 1 - 0.15 * index);
    const objective = Math.min(0.99, baseObjective + 0.05 * index);
    return {
      text: candidateText,
      objective: round(objective),
      coverage: round(coverage),
      selected: Math.round(coverage * survivors.length),
      chosen: false,
    };
  });
  let chosenIndex = 0;
  let best = -1;
  candidates.forEach((candidate, index) => {
    if (candidate.coverage >= 0.2 && candidate.objective > best) {
      best = candidate.objective;
      chosenIndex = index;
    }
  });
  candidates[chosenIndex].chosen = true;
  return {
    event: { type: "beam", beam_width: beamWidth, candidates, chosen_index: chosenIndex, refine_ms: 120 },
    winnerText: texts[chosenIndex],
  };
}

export async function deleteClauseMock(_clauseId: string): Promise<{ removed: boolean; refine_ms: number }> {
  return { removed: true, refine_ms: 4 };
}

function refineIntent(request: RefineRequest): { operation: ChipEvent["operation"]; text: string; confidence: number } {
  if (request.click) {
    return {
      operation: request.click.sign === "-" ? "exclude" : "require",
      text: `${request.click.sign === "-" ? "drop" : "keep"} ${request.click.chunk_id}`,
      confidence: 1,
    };
  }
  if (request.brush) {
    return { operation: "brush", text: `${request.brush.lo} to ${request.brush.hi}`, confidence: 1 };
  }
  const classified = classifyRefine(request.utterance ?? "");
  return { operation: classified.operation, text: request.utterance ?? "", confidence: classified.confidence };
}

function makeRefineDiff(request: RefineRequest, results: ResultEvent[]): Omit<DiffEvent, "refine_ms"> {
  if (request.click) {
    if (request.click.sign === "-") {
      return { type: "diff", added: [], removed: [request.click.chunk_id], rescored: [] };
    }
    return {
      type: "diff",
      added: [],
      removed: [],
      rescored: [{ chunk_id: request.click.chunk_id, score: 1 }],
    };
  }

  const classified = classifyRefine(request.utterance ?? "");
  const wantsPython = /\bpython\b/i.test(request.utterance ?? "");
  const removed: string[] = [];
  const rescored = results.map((result) => {
    let score = result.score;
    if (classified.operation === "exclude") score = result.score * 0.3;
    if (classified.operation === "require") score = wantsPython && result.meta.category !== "python" ? result.score * 0.35 : result.score * 0.96;
    if (classified.operation === "refocus") score = result.meta.type === "paper" ? Math.max(0.72, result.score) : result.score * 0.55;
    if (score < 0.5 && result.score >= 0.5) removed.push(result.chunk_id);
    return { chunk_id: result.chunk_id, score: Number(score.toFixed(3)) };
  });
  return { type: "diff", added: [], removed, rescored };
}

function applyMockDiff(results: ResultEvent[], diff: Omit<DiffEvent, "refine_ms">): ResultEvent[] {
  const removed = new Set(diff.removed);
  const rescored = new Map(diff.rescored.map((item) => [item.chunk_id, item.score]));
  return results
    .filter((result) => !removed.has(result.chunk_id))
    .map((result) => ({ ...result, score: rescored.get(result.chunk_id) ?? result.score }));
}

function makeAggregate(
  matched: ResultEvent[],
  threshold: number,
  scored: ResultEvent[] = mockResults(),
  corpus?: { total: number; scored: number; budget: number },
): AggregateEvent {
  return {
    type: "aggregate",
    scanned: scored.length,
    matched: matched.length,
    histogram: makeHistogram(scored),
    facets: {
      type: makeFacet("type", matched),
      category: makeFacet("category", matched),
      year: makeFacet("year", matched),
    },
    threshold,
    eta_ms: 0,
    corpus_total: corpus?.total ?? scored.length,
    corpus_scored: corpus?.scored ?? scored.length,
    compute_budget: corpus?.budget ?? 1,
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
  for (const result of mockResults()) {
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
