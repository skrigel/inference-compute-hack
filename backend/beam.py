"""Axis 3 (Truth): predicate beam search — the reusable core.

Extracted from ``backend.main`` so both the HTTP refine path and the agent-mode
tool layer (``backend.agent`` / ``backend.mcp_server``) drive one implementation.
The scorer, threshold, and minimum-coverage floor are injected rather than read
from module globals, which is what makes the agent layer testable in isolation.
"""

from __future__ import annotations

import time

from data.schema import Chunk
from inference.scorer import ScoreRequest, ScoreResult, ScorerClient

from backend.schemas import BeamCandidate, BeamEvent
from backend.select import facet_tokens


def candidate_predicates(
    utterance: str,
    survivors: set[str],
    chunks_by_id: dict[str, Chunk],
    beam_width: int,
) -> list[str]:
    """Generate a small candidate clause vocabulary for the beam.

    The raw utterance is always candidate 0; the rest are facet-narrowed variants
    derived from the facets most common among the current survivors.
    """
    base = utterance.strip()
    candidates = [base]
    counts: dict[str, int] = {}
    for chunk_id in survivors:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        for token in facet_tokens(chunk):
            counts[token] = counts.get(token, 0) + 1
    for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        name, _, value = token.partition(":")
        variant = f"{base} ({name} {value})"
        if variant not in candidates:
            candidates.append(variant)
        if len(candidates) >= beam_width:
            break
    return candidates[:beam_width]


async def score_candidate(
    scorer: ScorerClient,
    text: str,
    survivors: set[str],
    parent_scores: dict[str, ScoreResult],
    chunks_by_id: dict[str, Chunk],
    *,
    threshold: float,
) -> tuple[float, float, int]:
    """Evaluate one beam candidate. Returns ``(objective, coverage, selected)``.

    Objective = mean P(Yes) of the chunks that survive the candidate (require
    semantics: parent score x candidate score); coverage = fraction of parent
    survivors retained.
    """
    ids = [chunk_id for chunk_id in survivors if chunk_id in chunks_by_id]
    if not ids:
        return 0.0, 0.0, 0
    requests = [
        ScoreRequest(chunk_id=chunk_id, chunk_text=chunks_by_id[chunk_id].text, predicate=text)
        for chunk_id in ids
    ]
    selected_scores: list[float] = []
    for result in await scorer.score_batch(requests, tier=0):
        parent = parent_scores.get(result.chunk_id)
        combined = (parent.score if parent else 1.0) * result.score
        if combined >= threshold:
            selected_scores.append(combined)
    coverage = len(selected_scores) / len(ids)
    objective = sum(selected_scores) / len(selected_scores) if selected_scores else 0.0
    return objective, coverage, len(selected_scores)


async def run_beam(
    scorer: ScorerClient,
    utterance: str,
    beam_width: int,
    survivors: set[str],
    parent_scores: dict[str, ScoreResult],
    chunks_by_id: dict[str, Chunk],
    *,
    threshold: float,
    min_coverage: float,
) -> tuple[BeamEvent, str]:
    """Axis 3 (Truth): explore candidate clauses and objective-select the winner."""
    started = time.perf_counter()
    candidates = candidate_predicates(utterance, survivors, chunks_by_id, beam_width)
    evaluated: list[tuple[str, float, float, int]] = []
    for text in candidates:
        objective, coverage, selected = await score_candidate(
            scorer, text, survivors, parent_scores, chunks_by_id, threshold=threshold
        )
        evaluated.append((text, objective, coverage, selected))
    eligible = [item for item in evaluated if item[2] >= min_coverage]
    pool = eligible or evaluated
    chosen = max(pool, key=lambda item: (item[1], item[2]))
    chosen_index = evaluated.index(chosen)
    refine_ms = round((time.perf_counter() - started) * 1000.0)
    event = BeamEvent(
        beam_width=beam_width,
        candidates=[
            BeamCandidate(
                text=text,
                objective=objective,
                coverage=coverage,
                selected=selected,
                chosen=(index == chosen_index),
            )
            for index, (text, objective, coverage, selected) in enumerate(evaluated)
        ],
        chosen_index=chosen_index,
        refine_ms=refine_ms,
    )
    return event, chosen[0]
