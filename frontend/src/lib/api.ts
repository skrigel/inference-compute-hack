import { deleteClauseLive, ingestLive, queryLive, refineLive } from "./liveAdapter";
import { deleteClauseMock, ingestMock, queryMock, refineMock } from "./mockAdapter";
import type { Facets, FreshDocument, QueryEvent, QueryRequest, RefineEvent, RefineRequest } from "./types";

export type DataMode = "mock" | "live";

const MODE: DataMode = (import.meta.env.VITE_DATA_MODE ?? "mock") === "live" ? "live" : "mock";

export interface DashboardApi {
  mode: DataMode;
  ingest(corpusId: string, documents?: FreshDocument[]): Promise<{ n_chunks: number; facets: Facets }>;
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

// Live mode auto-falls-back to the mock adapter on any network error, so a
// backend outage degrades to the faithful mock instead of a blank screen.
export function createApi(): DashboardApi {
  if (MODE === "live") {
    return {
      mode: "live",
      async ingest(corpusId, documents) {
        try {
          return await ingestLive(corpusId, documents);
        } catch (error) {
          console.warn("live ingest failed; falling back to mock", error);
          return ingestMock(corpusId, documents);
        }
      },
      async query(request, onEvent, signal) {
        try {
          await queryLive(request, onEvent, signal);
        } catch (error) {
          if (isAbort(error)) return;
          console.warn("live query failed; falling back to mock", error);
          await queryViaMock(request, onEvent, signal);
        }
      },
      async refine(request, onEvent, signal) {
        try {
          await refineLive(request, onEvent, signal);
        } catch (error) {
          if (isAbort(error)) return;
          console.warn("live refine failed; falling back to mock", error);
          await refineViaMock(request, onEvent, signal);
        }
      },
      async deleteClause(clauseId) {
        try {
          return await deleteClauseLive(clauseId);
        } catch (error) {
          console.warn("live delete clause failed; falling back to mock", error);
          return deleteClauseMock(clauseId);
        }
      },
    };
  }

  return {
    mode: "mock",
    async ingest(corpusId, documents) {
      return ingestMock(corpusId, documents);
    },
    async query(request, onEvent, signal) {
      await queryViaMock(request, onEvent, signal);
    },
    async refine(request, onEvent, signal) {
      await refineViaMock(request, onEvent, signal);
    },
    async deleteClause(clauseId) {
      return deleteClauseMock(clauseId);
    },
  };
}

export const api = createApi();
