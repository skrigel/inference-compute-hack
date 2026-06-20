import { streamPost } from "./sse";
import type { Facets, QueryEvent, QueryRequest } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export async function ingestLive(corpusId: string): Promise<{ n_chunks: number; facets: Facets }> {
  const response = await fetch(`${API_BASE}/ingest`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ corpus_id: corpusId }),
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
