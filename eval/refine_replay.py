from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RefineTraceTurn:
    candidate_count: int
    chunks_scored: int
    survivor_count: int


def cumulative_curves(
    turns: list[RefineTraceTurn],
    *,
    n_chunks_total: int,
    rag_reindex_turns: set[int] | None = None,
) -> dict[str, list[int]]:
    """Counterfactual cumulative work curves from one measured scoped trace.

    The measured curve is ``chunks_scored``. The full-scan counterfactual scores
    the whole corpus every turn. The suffix-only counterfactual scores the same
    candidates but misses no score cache. The RAG baseline retrieves every turn
    and pays one extra corpus-sized re-index on configured 1-based turns.
    """
    reindex_turns = rag_reindex_turns or set()
    totals = {"scoped": 0, "full": 0, "suffix": 0, "rag": 0}
    curves: dict[str, list[int]] = {name: [] for name in totals}

    for index, turn in enumerate(turns, start=1):
        totals["scoped"] += turn.chunks_scored
        totals["full"] += n_chunks_total
        totals["suffix"] += turn.candidate_count
        totals["rag"] += n_chunks_total
        if index in reindex_turns:
            totals["rag"] += n_chunks_total
        for name, total in totals.items():
            curves[name].append(total)

    return curves
