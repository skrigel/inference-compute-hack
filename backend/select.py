"""Axis 2 (Movement): zero-inference selection over already-cached scores.

Both modes here are *pure arithmetic over cached per-chunk scores* — they never
touch the scorer. This is the "score first, then decide what to move" inversion
from the Infinite-Compute 3-Axis Framework:

- Mode A (threshold): auto-set the cutoff to a precision target — the smallest
  cutoff whose selected set still has mean P(Yes) >= target (maximises recall
  subject to the precision floor).
- Mode B (smart): max-coverage beam search over output subsets of size <= K.
  Each chunk covers the facet tokens implied by its metadata; we choose the
  subset that covers the most distinct facets within the movement budget K.
  Greedy gives the (1 - 1/e) submodular floor; widening the beam climbs toward
  the exhaustive optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

from data.schema import Chunk
from inference.scorer import ScoreResult


def auto_threshold(scores: list[float], precision_target: float) -> tuple[float, int]:
    """Return ``(threshold, selected_count)`` for Mode A.

    Because scores are taken best-first, the running mean is monotonically
    non-increasing, so there is a single crossover: the largest prefix whose
    mean P(Yes) is still >= ``precision_target``. The returned threshold is the
    lowest score included in that prefix.
    """
    ordered = sorted(scores, reverse=True)
    if not ordered:
        return 1.0, 0
    running = 0.0
    best_k = 0
    for k, score in enumerate(ordered, start=1):
        running += score
        if running / k >= precision_target:
            best_k = k
        else:
            break
    if best_k == 0:
        # Even the single best chunk is below the precision target: select none,
        # and set the cutoff just above the top score.
        return min(1.0, ordered[0] + 1e-9), 0
    return ordered[best_k - 1], best_k


def facet_tokens(chunk: Chunk) -> set[str]:
    """The distinct facet tokens a chunk "covers" for Mode B coverage."""
    tokens = {f"type:{chunk.type}"}
    meta = chunk.meta
    for name in ("category", "year", "repo", "lang"):
        value = getattr(meta, name, None)
        if value is not None:
            tokens.add(f"{name}:{value}")
    return tokens


@dataclass(frozen=True)
class SelectionResult:
    selected_ids: list[str]
    covered_facets: list[str]
    objective: float
    greedy_objective: float
    movement_budget: int
    beam_width: int
    candidate_pool: int


def _objective(covered: frozenset[str], score_sum: float) -> tuple[int, float]:
    """Lexicographic objective: facet coverage first, total score as tie-break."""
    return (len(covered), score_sum)


def max_coverage_select(
    items: list[tuple[str, set[str], float]],
    movement_budget: int,
    beam_width: int,
) -> tuple[list[str], frozenset[str], float, float]:
    """Beam search over subsets of size <= K maximising facet coverage.

    ``items`` is ``[(chunk_id, facet_tokens, score), ...]``. Returns
    ``(selected_ids, covered_tokens, objective_score, greedy_objective)`` where
    ``objective_score`` is the total score of the chosen subset and
    ``greedy_objective`` is the beam-width-1 score (the (1 - 1/e) floor).
    """
    budget = max(0, movement_budget)
    width = max(1, beam_width)
    if not items or budget == 0:
        return [], frozenset(), 0.0, 0.0

    def search(beam_size: int) -> tuple[list[str], frozenset[str], float]:
        # Each beam entry: (selected_ids tuple, covered frozenset, score_sum).
        beams: list[tuple[tuple[str, ...], frozenset[str], float]] = [((), frozenset(), 0.0)]
        best: tuple[tuple[str, ...], frozenset[str], float] = ((), frozenset(), 0.0)
        for _ in range(min(budget, len(items))):
            expanded: dict[frozenset[str], tuple[tuple[str, ...], frozenset[str], float]] = {}
            for selected, covered, score_sum in beams:
                selected_set = set(selected)
                for chunk_id, tokens, score in items:
                    if chunk_id in selected_set:
                        continue
                    new_selected = (*selected, chunk_id)
                    new_covered = covered | tokens
                    new_score = score_sum + score
                    key = frozenset(new_selected)
                    candidate = (new_selected, new_covered, new_score)
                    existing = expanded.get(key)
                    if existing is None or _objective(new_covered, new_score) > _objective(existing[1], existing[2]):
                        expanded[key] = candidate
            if not expanded:
                break
            ranked = sorted(
                expanded.values(),
                key=lambda entry: _objective(entry[1], entry[2]),
                reverse=True,
            )
            beams = ranked[:beam_size]
            if _objective(beams[0][1], beams[0][2]) > _objective(best[1], best[2]):
                best = beams[0]
        return list(best[0]), best[1], best[2]

    greedy_selected, greedy_covered, greedy_score = search(1)
    if width == 1:
        return greedy_selected, greedy_covered, greedy_score, greedy_score
    beam_selected, beam_covered, beam_score = search(width)
    # The beam can only improve on greedy under the lexicographic objective.
    if _objective(beam_covered, beam_score) >= _objective(greedy_covered, greedy_score):
        return beam_selected, beam_covered, beam_score, greedy_score
    return greedy_selected, greedy_covered, greedy_score, greedy_score


def smart_select(
    chunks_by_id: dict[str, Chunk],
    scores: dict[str, ScoreResult],
    *,
    threshold: float,
    movement_budget: int,
    beam_width: int,
) -> SelectionResult:
    """Mode B: choose <= K survivors that jointly cover the most facets."""
    pool = [
        (chunk_id, facet_tokens(chunks_by_id[chunk_id]), result.score)
        for chunk_id, result in scores.items()
        if chunk_id in chunks_by_id and result.score >= threshold
    ]
    pool.sort(key=lambda item: item[2], reverse=True)
    selected, covered, objective, greedy = max_coverage_select(pool, movement_budget, beam_width)
    return SelectionResult(
        selected_ids=selected,
        covered_facets=sorted(covered),
        objective=objective,
        greedy_objective=greedy,
        movement_budget=movement_budget,
        beam_width=beam_width,
        candidate_pool=len(pool),
    )
