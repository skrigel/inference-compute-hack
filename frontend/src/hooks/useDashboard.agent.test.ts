// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useDashboard } from "./useDashboard";

afterEach(() => {
  vi.restoreAllMocks();
});

// Guards the agent auto-pilot: one click must drive all three axes end-to-end
// (Memory scan -> Movement select -> Truth beam refine) and narrate each move.
describe("useDashboard — agent mode auto-pilot", () => {
  it("drives the three axes end-to-end and narrates each step", async () => {
    const { result } = renderHook(() => useDashboard());

    await act(async () => {
      await result.current.runAgent();
    });

    await waitFor(() => expect(result.current.agentRunning).toBe(false), { timeout: 12000 });

    // Narrated the four moves + the closing line.
    expect(result.current.agentLog.length).toBeGreaterThanOrEqual(5);
    expect(result.current.agentLog[result.current.agentLog.length - 1]).toMatch(/Done/);
    // Axis 1 scanned, Axis 2 left a selection, Axis 3 left a refine chip + beam.
    expect(result.current.hasRun).toBe(true);
    expect(result.current.selection).not.toBeNull();
    expect(result.current.chips.length).toBeGreaterThanOrEqual(1);
    expect(result.current.beamCandidates?.length ?? 0).toBeGreaterThan(0);
  }, 15000);
});
