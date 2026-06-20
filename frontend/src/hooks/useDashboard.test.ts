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

    const { result } = renderHook(() => useDashboard());

    // Run a query manually
    await act(async () => {
      await result.current.runQuery("retry network call without backoff");
    });
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

describe("useDashboard — Phase 2 refine loop", () => {
  it("runs natural-language refine without firing a new query", async () => {
    const querySpy = vi.spyOn(api, "query");

    const { result } = renderHook(() => useDashboard());
    await act(async () => {
      await result.current.runQuery("retry network call without backoff");
    });
    await waitFor(() => expect(result.current.streaming).toBe(false), { timeout: 4000 });
    querySpy.mockClear();

    await act(async () => {
      await result.current.runRefine("only python");
    });

    expect(querySpy).not.toHaveBeenCalled();
    expect(result.current.chips).toHaveLength(1);
    expect(result.current.chips[0].op).toBe("require");
    expect(result.current.latencyKind).toBe("warm");
  });

  it("removes a chip through the zero-inference clause deletion path", async () => {
    const deleteSpy = vi.spyOn(api, "deleteClause");

    const { result } = renderHook(() => useDashboard());
    await act(async () => {
      await result.current.runQuery("retry network call without backoff");
    });
    await waitFor(() => expect(result.current.streaming).toBe(false), { timeout: 4000 });

    await act(async () => {
      await result.current.runRefine("only python");
    });
    const chip = result.current.chips[0];

    await act(async () => {
      await result.current.removeChip(chip.clause_id);
    });

    expect(deleteSpy).toHaveBeenCalledWith(chip.clause_id);
    expect(result.current.chips).toHaveLength(0);
    expect(result.current.latencyKind).toBe("cached");
  });

  it("clears downstream dependent chips when an earlier chip is removed", async () => {
    const { result } = renderHook(() => useDashboard());
    await act(async () => {
      await result.current.runQuery("retry network call without backoff");
    });
    await waitFor(() => expect(result.current.streaming).toBe(false), { timeout: 4000 });

    await act(async () => {
      await result.current.runRefine("only python");
      await result.current.runRefine("without papers");
    });
    expect(result.current.chips).toHaveLength(2);

    await act(async () => {
      await result.current.removeChip(result.current.chips[0].clause_id);
    });

    expect(result.current.chips).toHaveLength(0);
    expect(result.current.latencyKind).toBe("cached");
  });

  it("ingests fresh files and automatically reruns the current query", async () => {
    const ingestSpy = vi.spyOn(api, "ingest");
    const querySpy = vi.spyOn(api, "query");

    const { result } = renderHook(() => useDashboard());
    await act(async () => {
      result.current.setPredicate("retry network call without backoff");
      await result.current.runQuery("retry network call without backoff");
    });
    await waitFor(() => expect(result.current.streaming).toBe(false), { timeout: 4000 });
    ingestSpy.mockClear();
    querySpy.mockClear();

    const file = new File(["fresh sentinel retry example without backoff"], "fresh_retry.py");
    await act(async () => {
      await result.current.ingestFreshFiles([file]);
    });

    expect(ingestSpy).toHaveBeenCalledWith(
      "demo",
      expect.arrayContaining([
        expect.objectContaining({ title: "fresh_retry.py", text: expect.stringContaining("sentinel") }),
      ]),
    );
    expect(querySpy).toHaveBeenCalledTimes(1);
  });
});
