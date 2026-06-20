// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../lib/api";
import { useDashboard } from "./useDashboard";

afterEach(() => {
  vi.restoreAllMocks();
});

// Guards the Axis-2 (Movement) selection wiring. The selection MATH lives in
// computeLab.test.ts; these tests cover the hook behaviour that two real bugs
// hid in: smart-select must move the slider to the survivor-pool edge (so the
// picked rows are visible) and the recut must stay zero-inference.
describe("useDashboard — Axis 2 selection wiring", () => {
  async function seeded() {
    const hook = renderHook(() => useDashboard());
    await act(async () => {
      await hook.result.current.runQuery("retry network call without backoff");
    });
    await waitFor(() => expect(hook.result.current.streaming).toBe(false), { timeout: 4000 });
    return hook.result;
  }

  it("smartSelect drops the slider to the pool edge and marks a smart selection (no scan)", async () => {
    const result = await seeded();
    const querySpy = vi.spyOn(api, "query");

    await act(async () => {
      result.current.smartSelect();
    });

    const selection = result.current.selection;
    expect(selection).not.toBeNull();
    expect(selection?.mode).toBe("smart");
    expect(selection!.selectedIds.length).toBeGreaterThan(0);
    // The fix: the threshold follows the survivor-pool boundary so the picked
    // (possibly lower-scoring, coverage-chosen) rows actually render.
    expect(result.current.threshold).toBeCloseTo(selection!.threshold, 6);
    // The beam objective can never be worse than the greedy floor.
    expect(selection!.objective).toBeGreaterThanOrEqual(selection!.greedyObjective);
    // Pure cache recut — selection never triggers inference.
    expect(querySpy).not.toHaveBeenCalled();
  });

  it("autoThreshold sets a threshold-mode selection and moves the slider", async () => {
    const result = await seeded();

    await act(async () => {
      result.current.autoThreshold();
    });

    const selection = result.current.selection;
    expect(selection?.mode).toBe("threshold");
    expect(result.current.threshold).toBeCloseTo(selection!.threshold, 6);
  });

  it("clearSelection removes the selection", async () => {
    const result = await seeded();
    await act(async () => {
      result.current.smartSelect();
    });
    expect(result.current.selection).not.toBeNull();

    await act(async () => {
      result.current.clearSelection();
    });
    expect(result.current.selection).toBeNull();
  });

  it("a fresh query clears any active selection", async () => {
    const result = await seeded();
    await act(async () => {
      result.current.smartSelect();
    });
    expect(result.current.selection).not.toBeNull();

    await act(async () => {
      await result.current.runQuery("exponential backoff jitter");
    });
    await waitFor(() => expect(result.current.streaming).toBe(false), { timeout: 4000 });
    expect(result.current.selection).toBeNull();
  });
});
