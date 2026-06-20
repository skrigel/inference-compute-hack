import { ingestLive, queryLive } from "./liveAdapter";
import { ingestMock, queryMock } from "./mockAdapter";
import type { Facets, QueryEvent, QueryRequest } from "./types";

export type DataMode = "mock" | "live";

const MODE: DataMode = (import.meta.env.VITE_DATA_MODE ?? "mock") === "live" ? "live" : "mock";

export interface DashboardApi {
  mode: DataMode;
  ingest(corpusId: string): Promise<{ n_chunks: number; facets: Facets }>;
  query(request: QueryRequest, onEvent: (event: QueryEvent) => void, signal?: AbortSignal): Promise<void>;
}

async function queryViaMock(
  request: QueryRequest,
  onEvent: (event: QueryEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  for await (const event of queryMock(request)) {
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
      async ingest(corpusId) {
        try {
          return await ingestLive(corpusId);
        } catch (error) {
          console.warn("live ingest failed; falling back to mock", error);
          return ingestMock(corpusId);
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
    };
  }

  return {
    mode: "mock",
    async ingest(corpusId) {
      return ingestMock(corpusId);
    },
    async query(request, onEvent, signal) {
      await queryViaMock(request, onEvent, signal);
    },
  };
}

export const api = createApi();
