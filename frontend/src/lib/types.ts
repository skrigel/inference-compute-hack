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
}

export type QueryEvent = ResultEvent | AggregateEvent | DoneEvent;
export type RefineEvent = ChipEvent | DiffEvent | AggregateEvent | DoneEvent;
export type StreamEvent = ResultEvent | AggregateEvent | DiffEvent | ChipEvent | DoneEvent;
