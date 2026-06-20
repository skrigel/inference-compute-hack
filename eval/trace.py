from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


CanonicalOperation = Literal[
    "query",
    "require",
    "exclude",
    "include",
    "refocus",
    "brush",
    "delete_clause",
]

OPERATION_ALIASES: dict[str, CanonicalOperation] = {
    "query": "query",
    "and": "require",
    "require": "require",
    "not": "exclude",
    "exclude": "exclude",
    "or": "include",
    "include": "include",
    "rewrite": "refocus",
    "refocus": "refocus",
    "threshold": "brush",
    "brush": "brush",
    "delete_clause": "delete_clause",
}


@dataclass(frozen=True)
class QualityMetrics:
    precision: float
    recall: float
    f1: float
    auc: float | None = None
    ece: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TurnTrace:
    run_id: str
    commit: str
    corpus_id: str
    model_id: str
    scorer_backend: str
    turn: int
    operation: str
    threshold: float
    n_chunks_total: int
    candidate_count: int
    chunks_scored: int
    chunks_served_from_cache: int
    survivor_count: int
    elapsed_ms: float
    model_ms: float
    queue_ms: float
    ttft_ms: float
    cache_hit_rate: float
    gpu_cache_usage_perc: float
    quality_slice: QualityMetrics | None = None

    @property
    def rho(self) -> float:
        if self.candidate_count == 0:
            return 0.0
        return self.survivor_count / self.candidate_count

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["operation"] = normalize_operation(self.operation)
        payload["rho"] = self.rho
        payload["quality_slice"] = (
            self.quality_slice.to_dict() if self.quality_slice is not None else None
        )
        return payload


def normalize_operation(operation: str) -> CanonicalOperation:
    normalized = operation.strip().lower()
    try:
        return OPERATION_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(OPERATION_ALIASES))
        raise ValueError(f"Unknown eval operation {operation!r}; expected one of: {allowed}") from exc
