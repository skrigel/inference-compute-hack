export const HIST_BINS = 20;

export const REFINE_OPS = ["require", "exclude", "include", "refocus", "brush"] as const;
export type RefineOp = (typeof REFINE_OPS)[number];

export interface ChunkMeta {
  type: "paper" | "code";
  title: string;
  category: string | null;
  year: number | null;
  path: string | null;
  lang: string | null;
  repo: string | null;
}

export interface FacetBucket {
  key: string;
  relevant: number;
  total: number;
}

export interface HistogramBin {
  lo: number;
  hi: number;
  count: number;
}

export interface Facets {
  type: FacetBucket[];
  category: FacetBucket[];
  year: FacetBucket[];
}

export interface QueryRequest {
  predicate: string;
  threshold: number;
  // Axis 1 (Memory): fraction of the corpus scored this query (0 < b <= 1).
  compute_budget?: number;
}

export interface FreshDocument {
  title: string;
  text: string;
  type: "paper" | "code";
  category: string | null;
  year: number | null;
  path: string | null;
  lang: string | null;
  repo: string | null;
}

export interface RefineRequest {
  utterance?: string;
  click?: {
    chunk_id: string;
    sign: "+" | "-";
  };
  brush?: {
    lo: number;
    hi: number;
  };
  // Axis 3 (Truth): 1 = single clause; >1 = explore N candidates and
  // objective-select the best.
  beam_width?: number;
}

export interface ResultEvent {
  type: "result";
  chunk_id: string;
  score: number;
  meta: ChunkMeta;
  rank: number;
  rationale: string | null;
}

export interface AggregateEvent {
  type: "aggregate";
  scanned: number;
  matched: number;
  histogram: HistogramBin[];
  facets: Facets;
  threshold: number;
  eta_ms: number;
  // Axis 1 (Memory): corpus scope for the active compute budget.
  corpus_total?: number;
  corpus_scored?: number;
  compute_budget?: number;
}

export interface Chip {
  clause_id: string;
  op: RefineOp;
  text: string;
  label: string;
  removable: boolean;
  confidence: number;
}

export interface ChipEvent {
  type: "chip";
  operation: RefineOp;
  chip: Chip;
  refine_ms: number;
  latency_kind: "cold" | "warm" | "cached";
}

export interface DiffEvent {
  type: "diff";
  added: ResultEvent[];
  removed: string[];
  rescored: Array<{
    chunk_id: string;
    score: number;
  }>;
  refine_ms: number;
}

export interface DoneEvent {
  type: "done";
  scanned: number;
  matched: number;
  elapsed_ms: number;
  warm: boolean;
  summary: string;
  // Axis 1 (Memory): corpus scope for the active compute budget.
  corpus_total?: number;
  corpus_scored?: number;
  compute_budget?: number;
}

export interface BeamCandidate {
  text: string;
  objective: number;
  coverage: number;
  selected: number;
  chosen: boolean;
}

export interface BeamEvent {
  type: "beam";
  beam_width: number;
  candidates: BeamCandidate[];
  chosen_index: number;
  refine_ms: number;
}

// Axis 2 (Movement): client-side selection over the complete score cache.
export type SelectMode = "threshold" | "smart";

export interface Selection {
  mode: SelectMode;
  threshold: number;
  selectedIds: string[];
  coveredFacets: string[];
  objective: number;
  greedyObjective: number;
  movementBudget: number;
  beamWidth: number;
  candidatePool: number;
}

export type QueryEvent = ResultEvent | AggregateEvent | DoneEvent;
export type RefineEvent = BeamEvent | ChipEvent | DiffEvent | AggregateEvent | DoneEvent;
export type StreamEvent = ResultEvent | AggregateEvent | DiffEvent | ChipEvent | DoneEvent | BeamEvent;

// Corpus management types
export interface Corpus {
  id: string;
  name: string;
  description: string;
  tags: string[];
  createdAt: number;
  lastUsedAt: number;
  isFavorite: boolean;
  isDemo: boolean;
  documentCount: number;
  source: "files" | "demo";
}

export interface SavedQuery {
  id: string;
  corpusId: string;
  predicate: string;
  threshold: number;
  chips: Chip[];
  name: string;
  notes: string;
  savedAt: number;
}
