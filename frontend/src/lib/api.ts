import { addArxivLive, deleteClauseLive, ingestLive, queryLive, refineLive } from "./liveAdapter";
import { addArxivMock, deleteClauseMock, ingestMock, queryMock, refineMock } from "./mockAdapter";
import type { Facets, FreshDocument, QueryEvent, QueryRequest, RefineEvent, RefineRequest } from "./types";

export type DataMode = "mock" | "live";

const STORAGE_KEY = "api-mode";

export function getApiMode(): DataMode {
  if (typeof window === "undefined") return "live";
  const stored = localStorage.getItem(STORAGE_KEY);
  // Default to live mode
  return stored === "mock" ? "mock" : "live";
}

export function setApiMode(mode: DataMode): void {
  localStorage.setItem(STORAGE_KEY, mode);
  window.dispatchEvent(new StorageEvent("storage", { key: STORAGE_KEY, newValue: mode }));
}

export interface DashboardApi {
  mode: DataMode;
  ingest(corpusId: string, documents?: FreshDocument[], limit?: number): Promise<{ n_chunks: number; facets: Facets }>;
  addArxiv(query: string, count?: number): Promise<{ n_chunks: number; facets: Facets }>;
  query(request: QueryRequest, onEvent: (event: QueryEvent) => void, signal?: AbortSignal): Promise<void>;
  refine(request: RefineRequest, onEvent: (event: RefineEvent) => void, signal?: AbortSignal): Promise<void>;
  deleteClause(clauseId: string): Promise<{ removed: boolean; refine_ms: number }>;
}

async function queryViaMock(
  request: QueryRequest,
  onEvent: (event: QueryEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  for await (const event of queryMock(request, signal)) {
    if (signal?.aborted) break;
    onEvent(event);
  }
}

async function refineViaMock(
  request: RefineRequest,
  onEvent: (event: RefineEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  for await (const event of refineMock(request, signal)) {
    if (signal?.aborted) break;
    onEvent(event);
  }
}

function isAbort(error: unknown): boolean {
  return typeof error === "object" && error !== null && (error as { name?: string }).name === "AbortError";
}

export function createApi(): DashboardApi {
  return {
    get mode(): DataMode {
      return getApiMode();
    },
    async ingest(corpusId, documents, limit) {
      if (getApiMode() === "live") {
        try {
          return await ingestLive(corpusId, documents, limit);
        } catch (error) {
          console.warn("live ingest failed; falling back to mock", error);
          return ingestMock(corpusId, documents, limit);
        }
      }
      return ingestMock(corpusId, documents, limit);
    },
    async addArxiv(query, count) {
      if (getApiMode() === "live") {
        try {
          return await addArxivLive(query, count);
        } catch (error) {
          console.warn("live arxiv ingest failed; falling back to mock", error);
          return addArxivMock(query, count);
        }
      }
      return addArxivMock(query, count);
    },
    async query(request, onEvent, signal) {
      if (getApiMode() === "live") {
        try {
          await queryLive(request, onEvent, signal);
          return;
        } catch (error) {
          if (isAbort(error)) return;
          console.warn("live query failed; falling back to mock", error);
        }
      }
      await queryViaMock(request, onEvent, signal);
    },
    async refine(request, onEvent, signal) {
      if (getApiMode() === "live") {
        try {
          await refineLive(request, onEvent, signal);
          return;
        } catch (error) {
          if (isAbort(error)) return;
          console.warn("live refine failed; falling back to mock", error);
        }
      }
      await refineViaMock(request, onEvent, signal);
    },
    async deleteClause(clauseId) {
      if (getApiMode() === "live") {
        try {
          return await deleteClauseLive(clauseId);
        } catch (error) {
          console.warn("live delete clause failed; falling back to mock", error);
          return deleteClauseMock(clauseId);
        }
      }
      return deleteClauseMock(clauseId);
    },
  };
}

export const api = createApi();
