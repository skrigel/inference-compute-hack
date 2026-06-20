import type { RefineOp } from "./types";

export interface ClassifiedRefine {
  operation: RefineOp;
  confidence: number;
}

const RULES: Array<{ operation: RefineOp; confidence: number; patterns: RegExp[] }> = [
  {
    operation: "brush",
    confidence: 0.9,
    patterns: [/\b(range|brush|drag|slider|threshold)\b/i, /\b\d(?:\.\d+)?\s*(to|-)\s*\d/i],
  },
  {
    operation: "refocus",
    confidence: 0.82,
    patterns: [/\b(i meant|actually|instead|refocus|focus on|in the .+ sense)\b/i],
  },
  {
    operation: "include",
    confidence: 0.8,
    patterns: [/\b(also include|include too|or|add back|show me too)\b/i],
  },
  {
    operation: "exclude",
    confidence: 0.84,
    patterns: [/\b(not|without|exclude|drop|remove|hide|minus|click minus)\b/i, /\bclick\s*-\b/i],
  },
  {
    operation: "require",
    confidence: 0.84,
    patterns: [/\b(only|must|require|requires|keep|plus|click plus)\b/i, /\bclick\s*\+\b/i],
  },
];

export function classifyRefine(input: string): ClassifiedRefine {
  for (const rule of RULES) {
    if (rule.patterns.some((pattern) => pattern.test(input))) {
      return { operation: rule.operation, confidence: rule.confidence };
    }
  }

  return { operation: "require", confidence: 0.5 };
}
