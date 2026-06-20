import { describe, expect, it } from "vitest";
import { classifyRefine } from "./classify";
import { REFINE_OPS } from "./types";

const allowedOps = new Set(REFINE_OPS);

describe("classifyRefine", () => {
  it.each([
    ["only in the networking layer", "require"],
    ["must mention retries", "require"],
    ["keep this result", "require"],
    ["not generated docs", "exclude"],
    ["without backoff", "exclude"],
    ["drop this result", "exclude"],
    ["also include timeout handling", "include"],
    ["or connection pooling", "include"],
    ["I meant exponential backoff", "refocus"],
    ["actually focus on http clients", "refocus"],
    ["range 0.6 to 1.0", "brush"],
    ["drag threshold above 0.7", "brush"],
  ] as const)("maps %s to %s", (input, op) => {
    expect(classifyRefine(input).operation).toBe(op);
  });

  it("never emits an operation outside the contract vocabulary", () => {
    const samples = [
      "",
      "something ambiguous",
      "only code without comments",
      "actually also include papers",
      "click plus",
      "click minus",
    ];

    for (const sample of samples) {
      expect(allowedOps.has(classifyRefine(sample).operation)).toBe(true);
    }
  });
});
