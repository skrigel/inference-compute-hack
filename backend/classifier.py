from __future__ import annotations

import re
from dataclasses import dataclass

from backend.schemas import RefineOp


@dataclass(frozen=True)
class ClassifiedRefine:
    operation: RefineOp
    confidence: float


RULES: tuple[tuple[RefineOp, float, tuple[re.Pattern[str], ...]], ...] = (
    (
        RefineOp.brush,
        0.9,
        (
            re.compile(r"\b(range|brush|drag|slider|threshold)\b", re.I),
            re.compile(r"\b\d(?:\.\d+)?\s*(to|-)\s*\d", re.I),
        ),
    ),
    (
        RefineOp.refocus,
        0.82,
        (re.compile(r"\b(i meant|actually|instead|refocus|focus on|in the .+ sense)\b", re.I),),
    ),
    (
        RefineOp.include,
        0.8,
        (re.compile(r"\b(also include|include too|or|add back|show me too)\b", re.I),),
    ),
    (
        RefineOp.exclude,
        0.84,
        (
            re.compile(r"\b(not|without|exclude|drop|remove|hide|minus|click minus)\b", re.I),
            re.compile(r"\bclick\s*-\b", re.I),
        ),
    ),
    (
        RefineOp.require,
        0.84,
        (
            re.compile(r"\b(only|must|require|requires|keep|plus|click plus)\b", re.I),
            re.compile(r"\bclick\s*\+\b", re.I),
        ),
    ),
)


def classify_refine(text: str) -> ClassifiedRefine:
    for operation, confidence, patterns in RULES:
        if any(pattern.search(text) for pattern in patterns):
            return ClassifiedRefine(operation, confidence)
    return ClassifiedRefine(RefineOp.require, 0.5)

