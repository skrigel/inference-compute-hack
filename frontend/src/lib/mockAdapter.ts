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
const LARGE_STREAM_DELAY_MS = 0;

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

// BrowseComp+ mock items - simulated web documents.
function browseCompItems(limit = 100): MockItem[] {
  return Array.from({ length: Math.max(1, limit) }, (_, i) => ({
    chunk_id: `bcp-${i}`,
    score: Math.max(0.05, 0.95 - (i * 0.009) + (Math.sin(i) * 0.1)),
    type: "paper" as const,
    title: `BrowseComp Document ${i + 1}`,
    category: ["news", "wiki", "blog", "academic", "forum"][i % 5],
    year: 2020 + (i % 6),
    path: `https://example.com/doc/${i}`,
    repo: null,
  }));
}

// arXiv ML papers mock items
const ARXIV_ML_ITEMS: MockItem[] = [
  { chunk_id: "arxiv-01", score: 0.94, type: "paper", title: "Attention Is All You Need", category: "cs.CL", year: 2017, path: "arxiv:1706.03762", repo: null },
  { chunk_id: "arxiv-02", score: 0.92, type: "paper", title: "BERT: Pre-training of Deep Bidirectional Transformers", category: "cs.CL", year: 2018, path: "arxiv:1810.04805", repo: null },
  { chunk_id: "arxiv-03", score: 0.89, type: "paper", title: "Language Models are Few-Shot Learners (GPT-3)", category: "cs.CL", year: 2020, path: "arxiv:2005.14165", repo: null },
  { chunk_id: "arxiv-04", score: 0.87, type: "paper", title: "LLaMA: Open and Efficient Foundation Language Models", category: "cs.CL", year: 2023, path: "arxiv:2302.13971", repo: null },
  { chunk_id: "arxiv-05", score: 0.85, type: "paper", title: "FlashAttention: Fast and Memory-Efficient Attention", category: "cs.LG", year: 2022, path: "arxiv:2205.14135", repo: null },
  { chunk_id: "arxiv-06", score: 0.83, type: "paper", title: "Scaling Laws for Neural Language Models", category: "cs.LG", year: 2020, path: "arxiv:2001.08361", repo: null },
  { chunk_id: "arxiv-07", score: 0.81, type: "paper", title: "Training Compute-Optimal Large Language Models", category: "cs.LG", year: 2022, path: "arxiv:2203.15556", repo: null },
  { chunk_id: "arxiv-08", score: 0.78, type: "paper", title: "Constitutional AI: Harmlessness from AI Feedback", category: "cs.CL", year: 2022, path: "arxiv:2212.08073", repo: null },
  { chunk_id: "arxiv-09", score: 0.75, type: "paper", title: "Chain-of-Thought Prompting Elicits Reasoning", category: "cs.CL", year: 2022, path: "arxiv:2201.11903", repo: null },
  { chunk_id: "arxiv-10", score: 0.72, type: "paper", title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP", category: "cs.CL", year: 2020, path: "arxiv:2005.11401", repo: null },
  { chunk_id: "arxiv-11", score: 0.69, type: "paper", title: "LoRA: Low-Rank Adaptation of Large Language Models", category: "cs.CL", year: 2021, path: "arxiv:2106.09685", repo: null },
  { chunk_id: "arxiv-12", score: 0.66, type: "paper", title: "Speculative Decoding for LLM Inference", category: "cs.LG", year: 2023, path: "arxiv:2302.01318", repo: null },
  { chunk_id: "arxiv-13", score: 0.63, type: "paper", title: "GPTQ: Accurate Post-Training Quantization", category: "cs.LG", year: 2022, path: "arxiv:2210.17323", repo: null },
  { chunk_id: "arxiv-14", score: 0.60, type: "paper", title: "AWQ: Activation-aware Weight Quantization", category: "cs.LG", year: 2023, path: "arxiv:2306.00978", repo: null },
  { chunk_id: "arxiv-15", score: 0.57, type: "paper", title: "vLLM: Efficient Memory Management for LLM Serving", category: "cs.LG", year: 2023, path: "arxiv:2309.06180", repo: null },
  { chunk_id: "arxiv-16", score: 0.54, type: "paper", title: "Mixture of Experts Meets Instruction Tuning", category: "cs.CL", year: 2023, path: "arxiv:2305.14705", repo: null },
  { chunk_id: "arxiv-17", score: 0.51, type: "paper", title: "Direct Preference Optimization (DPO)", category: "cs.LG", year: 2023, path: "arxiv:2305.18290", repo: null },
  { chunk_id: "arxiv-18", score: 0.48, type: "paper", title: "Toolformer: Language Models Can Teach Themselves to Use Tools", category: "cs.CL", year: 2023, path: "arxiv:2302.04761", repo: null },
  { chunk_id: "arxiv-19", score: 0.45, type: "paper", title: "Self-Instruct: Aligning LMs with Self-Generated Instructions", category: "cs.CL", year: 2022, path: "arxiv:2212.10560", repo: null },
  { chunk_id: "arxiv-20", score: 0.42, type: "paper", title: "Prefix Caching for Efficient LLM Serving", category: "cs.LG", year: 2024, path: "arxiv:2401.xxxxx", repo: null },
  { chunk_id: "arxiv-21", score: 0.39, type: "paper", title: "KV Cache Compression for Long Context", category: "cs.LG", year: 2024, path: "arxiv:2402.xxxxx", repo: null },
  { chunk_id: "arxiv-22", score: 0.36, type: "paper", title: "Ring Attention for Long Sequences", category: "cs.LG", year: 2023, path: "arxiv:2310.01889", repo: null },
  { chunk_id: "arxiv-23", score: 0.33, type: "paper", title: "Continuous Batching for LLM Inference", category: "cs.DC", year: 2023, path: "arxiv:2309.xxxxx", repo: null },
  { chunk_id: "arxiv-24", score: 0.30, type: "paper", title: "FP8 Training and Inference for Transformers", category: "cs.LG", year: 2023, path: "arxiv:2310.xxxxx", repo: null },
  { chunk_id: "arxiv-25", score: 0.27, type: "paper", title: "PagedAttention for Dynamic Memory Allocation", category: "cs.LG", year: 2023, path: "arxiv:2309.06180", repo: null },
];

// Open source codebase mock items
const CODEBASE_ITEMS: MockItem[] = [
  { chunk_id: "code-01", score: 0.95, type: "code", title: "requests/sessions.py", category: "python", year: 2023, path: "requests/sessions.py", repo: "requests" },
  { chunk_id: "code-02", score: 0.92, type: "code", title: "httpx/_client.py", category: "python", year: 2024, path: "httpx/_client.py", repo: "httpx" },
  { chunk_id: "code-03", score: 0.89, type: "code", title: "aiohttp/connector.py", category: "python", year: 2023, path: "aiohttp/connector.py", repo: "aiohttp" },
  { chunk_id: "code-04", score: 0.86, type: "code", title: "grpc-go/balancer/roundrobin.go", category: "go", year: 2023, path: "balancer/roundrobin/roundrobin.go", repo: "grpc-go" },
  { chunk_id: "code-05", score: 0.83, type: "code", title: "tokio/runtime/scheduler.rs", category: "rust", year: 2024, path: "tokio/src/runtime/scheduler/mod.rs", repo: "tokio" },
  { chunk_id: "code-06", score: 0.80, type: "code", title: "fastapi/routing.py", category: "python", year: 2024, path: "fastapi/routing.py", repo: "fastapi" },
  { chunk_id: "code-07", score: 0.77, type: "code", title: "pydantic/main.py", category: "python", year: 2024, path: "pydantic/main.py", repo: "pydantic" },
  { chunk_id: "code-08", score: 0.74, type: "code", title: "sqlalchemy/engine/base.py", category: "python", year: 2023, path: "lib/sqlalchemy/engine/base.py", repo: "sqlalchemy" },
  { chunk_id: "code-09", score: 0.71, type: "code", title: "kubernetes/client-go/rest/request.go", category: "go", year: 2023, path: "rest/request.go", repo: "client-go" },
  { chunk_id: "code-10", score: 0.68, type: "code", title: "reqwest/src/async_impl/client.rs", category: "rust", year: 2024, path: "src/async_impl/client.rs", repo: "reqwest" },
  { chunk_id: "code-11", score: 0.65, type: "code", title: "celery/app/task.py", category: "python", year: 2023, path: "celery/app/task.py", repo: "celery" },
  { chunk_id: "code-12", score: 0.62, type: "code", title: "redis-py/redis/client.py", category: "python", year: 2024, path: "redis/client.py", repo: "redis-py" },
  { chunk_id: "code-13", score: 0.59, type: "code", title: "prometheus/client_golang/prometheus.go", category: "go", year: 2023, path: "prometheus/prometheus.go", repo: "client_golang" },
  { chunk_id: "code-14", score: 0.56, type: "code", title: "hyper/src/client/conn.rs", category: "rust", year: 2024, path: "src/client/conn.rs", repo: "hyper" },
  { chunk_id: "code-15", score: 0.53, type: "code", title: "boto3/resources/action.py", category: "python", year: 2023, path: "boto3/resources/action.py", repo: "boto3" },
  { chunk_id: "code-16", score: 0.50, type: "code", title: "django/db/backends/base/base.py", category: "python", year: 2024, path: "django/db/backends/base/base.py", repo: "django" },
  { chunk_id: "code-17", score: 0.47, type: "code", title: "gin-gonic/gin/context.go", category: "go", year: 2023, path: "context.go", repo: "gin" },
  { chunk_id: "code-18", score: 0.44, type: "code", title: "actix-web/src/server.rs", category: "rust", year: 2024, path: "actix-web/src/server.rs", repo: "actix-web" },
  { chunk_id: "code-19", score: 0.41, type: "code", title: "numpy/core/numeric.py", category: "python", year: 2023, path: "numpy/core/numeric.py", repo: "numpy" },
  { chunk_id: "code-20", score: 0.38, type: "code", title: "pandas/core/frame.py", category: "python", year: 2024, path: "pandas/core/frame.py", repo: "pandas" },
  { chunk_id: "code-21", score: 0.35, type: "code", title: "etcd/client/v3/client.go", category: "go", year: 2023, path: "client/v3/client.go", repo: "etcd" },
  { chunk_id: "code-22", score: 0.32, type: "code", title: "serde/src/de/mod.rs", category: "rust", year: 2024, path: "serde/src/de/mod.rs", repo: "serde" },
  { chunk_id: "code-23", score: 0.29, type: "code", title: "pytorch/torch/nn/modules/module.py", category: "python", year: 2024, path: "torch/nn/modules/module.py", repo: "pytorch" },
  { chunk_id: "code-24", score: 0.26, type: "code", title: "transformers/modeling_utils.py", category: "python", year: 2024, path: "src/transformers/modeling_utils.py", repo: "transformers" },
  { chunk_id: "code-25", score: 0.23, type: "code", title: "vllm/engine/llm_engine.py", category: "python", year: 2024, path: "vllm/engine/llm_engine.py", repo: "vllm" },
];

type CorpusId = "demo" | "browsecomp" | "arxiv-ml" | "codebase";
let activeCorpus: CorpusId = "demo";
let activeBrowseCompLimit = 100;
let freshItems: MockItem[] = [];
let clauseSeq = 1;

function getBaseItems(): MockItem[] {
  switch (activeCorpus) {
    case "browsecomp":
      return browseCompItems(activeBrowseCompLimit);
    case "arxiv-ml":
      return ARXIV_ML_ITEMS;
    case "codebase":
      return CODEBASE_ITEMS;
    default:
      return DEMO_ITEMS;
  }
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
  limit?: number,
): Promise<{ n_chunks: number; facets: Facets }> {
  // Switch corpus based on corpusId
  const validCorpora: CorpusId[] = ["demo", "browsecomp", "arxiv-ml", "codebase"];
  if (validCorpora.includes(corpusId as CorpusId)) {
    activeCorpus = corpusId as CorpusId;
    if (corpusId === "browsecomp") activeBrowseCompLimit = limit ?? activeBrowseCompLimit;
    if (!documents.length) freshItems = [];
  }
  if (documents.length > 0) {
    const nextItems = documents.map((document, index) => ({
      chunk_id: `fresh-${Date.now()}-${index}`,
      score: document.text.toLowerCase().includes("sentinel") || document.repo === "arxiv" ? 0.97 : 0.72,
      type: document.type,
      title: document.title,
      category: document.category ?? document.lang ?? "fresh",
      year: document.year ?? new Date().getFullYear(),
      path: document.path,
      repo: document.repo,
    }));
    freshItems = [...freshItems, ...nextItems];
  }
  return { n_chunks: mockResults().length, facets: allFacets() };
}

export async function addArxivMock(query: string, count = 25): Promise<{ n_chunks: number; facets: Facets }> {
  const total = Math.max(1, Math.min(100, Math.round(count)));
  const documents: FreshDocument[] = Array.from({ length: total }, (_, index) => ({
    title: `arXiv ${query} ${index + 1}`,
    text: `${query} synthetic arxiv paper about reward variance, retrieval ranking, and verifier-backed evidence ${index + 1}`,
    type: "paper",
    category: "cs.IR",
    year: new Date().getFullYear(),
    path: `arxiv:demo.${Date.now()}.${index}`,
    lang: null,
    repo: "arxiv",
  }));
  return ingestMock(activeCorpus, documents);
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
  const streamDelay = results.length > 200 ? LARGE_STREAM_DELAY_MS : STREAM_DELAY_MS;
  for (const result of results) {
    if (signal?.aborted) return;
    yield result;
    if (streamDelay) await sleep(streamDelay);
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
