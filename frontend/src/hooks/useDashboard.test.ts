// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../lib/api";
import { useDashboard } from "./useDashboard";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useDashboard — threshold drag is zero-inference (Phase 1 exit gate)", () => {
  it("dragging the threshold through the hook never calls api.query or api.ingest", async () => {
    const querySpy = vi.spyOn(api, "query");
    const ingestSpy = vi.spyOn(api, "ingest");

    const { result } = renderHook(() => useDashboard("retry network call without backoff"));

    // Let the seed query stream to completion.
    await waitFor(() => expect(result.current.streaming).toBe(false), { timeout: 4000 });
    expect(result.current.hasRun).toBe(true);

    // From here on, only threshold drags happen — no scan, no refine.
    querySpy.mockClear();
    ingestSpy.mockClear();

    for (let threshold = 0; threshold <= 1.0001; threshold += 0.1) {
      await act(async () => {
        result.current.setThreshold(threshold);
      });
    }

    expect(querySpy).not.toHaveBeenCalled();
    expect(ingestSpy).not.toHaveBeenCalled();

    // And the recut actually responded: a low threshold matches more than a high one.
    await act(async () => result.current.setThreshold(0.1));
    const lowMatched = result.current.view.matched;
    await act(async () => result.current.setThreshold(0.9));
    const highMatched = result.current.view.matched;
    expect(lowMatched).toBeGreaterThan(highMatched);
  });
});
