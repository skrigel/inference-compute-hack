// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../lib/api";
import { useDashboard } from "./useDashboard";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useDashboard — MCP comparison demo", () => {
  it("changes the active demo dataset size through corpus ingestion", async () => {
    const ingestSpy = vi.spyOn(api, "ingest");
    const { result } = renderHook(() => useDashboard());

    await act(async () => {
      await result.current.setDemoDatasetSize(1000);
    });

    expect(result.current.demoDatasetSize).toBe(1000);
    expect(ingestSpy).toHaveBeenCalledWith("browsecomp", [], 1000);
  });

  it("runs an iterative task and reports ours-vs-rag speedup", async () => {
    const { result } = renderHook(() => useDashboard());

    await act(async () => {
      await result.current.runComparisonTask("rl-metrics");
    });

    await waitFor(() => expect(result.current.comparisonRunning).toBe(false), { timeout: 15000 });

    const comparison = result.current.comparison;
    expect(comparison).not.toBeNull();
    expect(comparison?.taskId).toBe("rl-metrics");
    expect(comparison?.ours.toolName).toBe("search_source_ours");
    expect(comparison?.rag.toolName).toBe("search_source_rag");
    expect(comparison?.speedup).toBeGreaterThan(1);
    expect(comparison?.ours.steps.length).toBeGreaterThan(comparison?.rag.steps.length ?? 0);
  }, 16000);

  it("adds an arxiv burst through the source API and reruns the current query", async () => {
    const arxivSpy = vi.spyOn(api, "addArxiv");
    const querySpy = vi.spyOn(api, "query");
    const { result } = renderHook(() => useDashboard());

    await act(async () => {
      result.current.setPredicate("retrieval ranking metrics");
      await result.current.runQuery("retrieval ranking metrics");
    });
    await waitFor(() => expect(result.current.streaming).toBe(false), { timeout: 4000 });
    arxivSpy.mockClear();
    querySpy.mockClear();

    await act(async () => {
      await result.current.addArxivBurst("retrieval ranking metrics", 8);
    });

    expect(arxivSpy).toHaveBeenCalledWith("retrieval ranking metrics", 8);
    expect(result.current.dynamicSourceCount).toBeGreaterThanOrEqual(8);
    expect(querySpy).toHaveBeenCalledTimes(1);
  });
});
