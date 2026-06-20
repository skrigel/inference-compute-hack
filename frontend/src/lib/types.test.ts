import { describe, expect, it } from "vitest";
import { HIST_BINS, REFINE_OPS } from "./types";

describe("contract types", () => {
  it("exports the frozen histogram bin count", () => {
    expect(HIST_BINS).toBe(20);
  });

  it("exports exactly the allowed refine operations", () => {
    expect(REFINE_OPS).toEqual([
      "require",
      "exclude",
      "include",
      "refocus",
      "brush",
    ]);
  });
});
