import { streamPost } from "./sse";
import type { Facets, FreshDocument, QueryEvent, QueryRequest, RefineEvent, RefineRequest } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export async function ingestLive(
  corpusId: string,
  documents: FreshDocument[] = [],
  limit?: number,
): Promise<{ n_chunks: number; facets: Facets }> {
  const response = await fetch(`${API_BASE}/ingest`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ corpus_id: corpusId, documents, limit }),
  });
  if (!response.ok) throw new Error(`ingest failed: ${response.status}`);
  const data = await response.json();
  return { n_chunks: data.n_chunks, facets: data.facets };
}

export async function queryLive(
  request: QueryRequest,
  onEvent: (event: QueryEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  await streamPost(`${API_BASE}/query`, request, (event) => onEvent(event as QueryEvent), signal);
}

export async function refineLive(
  request: RefineRequest,
  onEvent: (event: RefineEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  await streamPost(`${API_BASE}/refine`, request, (event) => onEvent(event as RefineEvent), signal);
}

export async function deleteClauseLive(clauseId: string): Promise<{ removed: boolean; refine_ms: number }> {
  const response = await fetch(`${API_BASE}/clause/${encodeURIComponent(clauseId)}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error(`delete clause failed: ${response.status}`);
  return response.json();
}
