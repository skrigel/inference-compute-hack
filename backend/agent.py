"""Agent mode — the instance-scoped tool layer over the three compute axes.

This is the framework an autonomous agent (or the MCP server in
``backend.mcp_server``) drives instead of the human UI. Each ``AgentSession``
owns its own corpus state + score cache + scorer, and exposes the three axes as
plain async methods returning JSON-friendly dicts:

- Axis 1 (Memory):   ``query(predicate, compute_budget=...)`` scopes how much of
  the corpus is scored.
- Axis 2 (Movement): ``select(mode=...)`` auto-thresholds or max-coverage
  smart-selects over the cached scores (zero inference).
- Axis 3 (Truth):    ``refine(utterance, beam_width=...)`` runs the predicate
  beam and applies the objective-selected winner.

The HTTP app (``backend.main``) keeps its own module-level singletons; this
layer is deliberately separate so an agent can hold many independent sessions
and so it is unit-testable with a scripted scorer.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import replace

from data.schema import Chunk
from inference.config import make_scorer
from inference.scorer import ScoreRequest, ScoreResult, ScorerClient

from backend.beam import run_beam
from backend.cache import ScoreCache
from backend.select import auto_threshold, smart_select
from backend.state import BackendState
from backend.streaming import query_stream

# Mirror backend.main: a beam candidate must retain this fraction of the parent
# survivors to be eligible for the objective max.
BEAM_MIN_COVERAGE = float(os.environ.get("BEAM_MIN_COVERAGE", "0.2"))


def _result_dict(result: ScoreResult, chunk: Chunk, rank: int) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "rank": rank,
        "score": round(result.score, 4),
        "type": chunk.type,
        "title": chunk.title,
        "category": chunk.meta.category,
        "year": chunk.meta.year,
        "repo": chunk.meta.repo,
        "lang": chunk.meta.lang,
        "path": chunk.meta.path,
    }


class AgentSession:
    """A single agent's view of one corpus across the three axes."""

    def __init__(self, scorer: ScorerClient | None = None) -> None:
        self.scorer = scorer or make_scorer()
        self.state = BackendState()
        self.cache = ScoreCache()
        self.threshold = 0.5
        self._seq = itertools.count(1)

    # -- corpus -----------------------------------------------------------------
    async def ingest(self, corpus_id: str = "demo", *, limit: int | None = None) -> dict:
        """Load a corpus into the session. Resets any prior query/clause state."""
        if corpus_id == "browsecomp":
            self.state.load_browsecomp(limit=limit)
        else:
            corpus_id = "demo"
            self.state.load_demo()
        self.cache.clear()
        self.state.current_clause = None
        return {"corpus_id": corpus_id, "n_chunks": len(self.state.chunks)}

    # -- Axis 1 (Memory) --------------------------------------------------------
    async def query(
        self,
        predicate: str,
        *,
        compute_budget: float = 1.0,
        threshold: float = 0.5,
        top_k: int | None = None,
    ) -> dict:
        """Score the budgeted slice of the corpus and return ranked survivors."""
        if not self.state.chunks:
            self.state.load_demo()
        clause_id = f"q{next(self._seq)}"
        self.state.current_clause = clause_id
        self.state.threshold = threshold
        self.threshold = threshold

        async for _ in query_stream(
            self.scorer,
            self.state.chunks,
            predicate,
            clause_id=clause_id,
            threshold=threshold,
            cache=self.cache,
            compute_budget=compute_budget,
        ):
            pass  # query_stream populates the cache; we read it back below.

        scored = self.cache.scores_for_clause(clause_id)
        return {
            "clause_id": clause_id,
            "predicate": predicate,
            "compute_budget": compute_budget,
            "threshold": threshold,
            "corpus_total": len(self.state.chunks),
            "corpus_scored": len(scored),
            **self._slice(scored, threshold=threshold, top_k=top_k),
        }

    # -- Axis 2 (Movement) ------------------------------------------------------
    def select(
        self,
        *,
        mode: str = "threshold",
        precision_target: float = 0.85,
        movement_budget: int = 5,
        beam_width: int = 4,
    ) -> dict:
        """Auto-threshold (Mode A) or max-coverage smart-select (Mode B).

        Pure arithmetic over the cached scores — never calls the scorer.
        """
        scored = self._current_scores()
        chunks_by_id = self.state.chunks_by_id()
        score_values = [r.score for cid, r in scored.items() if cid in chunks_by_id]
        threshold, _ = auto_threshold(score_values, precision_target)

        if mode != "smart":
            selected = sorted(
                (cid for cid, r in scored.items() if cid in chunks_by_id and r.score >= threshold),
                key=lambda cid: scored[cid].score,
                reverse=True,
            )
            return {
                "mode": "threshold",
                "threshold": round(threshold, 4),
                "selected_ids": selected,
                "selected_count": len(selected),
            }

        selection = smart_select(
            chunks_by_id,
            scored,
            threshold=threshold,
            movement_budget=movement_budget,
            beam_width=beam_width,
        )
        return {
            "mode": "smart",
            "threshold": round(threshold, 4),
            "selected_ids": selection.selected_ids,
            "selected_count": len(selection.selected_ids),
            "covered_facets": selection.covered_facets,
            "objective": round(selection.objective, 4),
            "greedy_objective": round(selection.greedy_objective, 4),
            "movement_budget": selection.movement_budget,
            "beam_width": selection.beam_width,
            "candidate_pool": selection.candidate_pool,
        }

    # -- Axis 3 (Truth) ---------------------------------------------------------
    async def refine(self, utterance: str, *, beam_width: int = 4) -> dict:
        """Run the predicate beam and apply the objective-selected winner clause."""
        parent_clause = self.state.current_clause
        if parent_clause is None:
            raise ValueError("Run query() before refine()")
        parent_scores = self.cache.scores_for_clause(parent_clause)
        if not parent_scores:
            raise ValueError("Current query has no cached scores yet")
        chunks_by_id = self.state.chunks_by_id()
        survivors = {cid for cid, r in parent_scores.items() if r.score >= self.threshold}

        beam_event, winner_text = await run_beam(
            self.scorer,
            utterance,
            max(1, beam_width),
            survivors,
            parent_scores,
            chunks_by_id,
            threshold=self.threshold,
            min_coverage=BEAM_MIN_COVERAGE,
        )

        clause_id = f"c{next(self._seq)}"
        next_scores = await self._apply_require(
            winner_text, survivors, parent_scores, chunks_by_id, clause_id
        )
        self.state.current_clause = clause_id

        chosen = beam_event.candidates[beam_event.chosen_index]
        return {
            "clause_id": clause_id,
            "utterance": utterance,
            "chosen": winner_text,
            "beam_width": beam_event.beam_width,
            "objective": round(chosen.objective, 4),
            "candidates": [
                {
                    "text": c.text,
                    "objective": round(c.objective, 4),
                    "coverage": round(c.coverage, 4),
                    "selected": c.selected,
                    "chosen": c.chosen,
                }
                for c in beam_event.candidates
            ],
            **self._slice(next_scores, threshold=self.threshold, top_k=None),
        }

    # -- read-only --------------------------------------------------------------
    def results(self, *, threshold: float | None = None, top_k: int | None = None) -> dict:
        """Ranked slice of the current clause — a pure cache read."""
        cut = self.threshold if threshold is None else threshold
        return self._slice(self._current_scores(), threshold=cut, top_k=top_k)

    # -- internals --------------------------------------------------------------
    def _current_scores(self) -> dict[str, ScoreResult]:
        clause = self.state.current_clause
        return self.cache.scores_for_clause(clause) if clause else {}

    def _slice(self, scored: dict[str, ScoreResult], *, threshold: float, top_k: int | None) -> dict:
        chunks_by_id = self.state.chunks_by_id()
        ranked = sorted(
            (r for cid, r in scored.items() if cid in chunks_by_id),
            key=lambda r: r.score,
            reverse=True,
        )
        matched = [r for r in ranked if r.score >= threshold]
        sliced = matched[:top_k] if top_k is not None else matched
        return {
            "matched": len(matched),
            "results": [
                _result_dict(r, chunks_by_id[r.chunk_id], rank) for rank, r in enumerate(sliced)
            ],
        }

    async def _apply_require(
        self,
        text: str,
        survivors: set[str],
        parent_scores: dict[str, ScoreResult],
        chunks_by_id: dict[str, Chunk],
        clause_id: str,
    ) -> dict[str, ScoreResult]:
        """Rescore the survivors with the winner clause (require: parent x candidate)."""
        ids = [cid for cid in survivors if cid in chunks_by_id]
        requests = [
            ScoreRequest(chunk_id=cid, chunk_text=chunks_by_id[cid].text, predicate=text)
            for cid in ids
        ]
        next_scores: dict[str, ScoreResult] = {}
        for result in await self.scorer.score_batch(requests, tier=0):
            parent = parent_scores.get(result.chunk_id)
            combined = (parent.score if parent else 1.0) * result.score
            rescored = replace(result, score=combined, p_yes=combined, p_no=1.0 - combined)
            self.cache.put(result.chunk_id, clause_id, rescored)
            next_scores[result.chunk_id] = rescored
        return next_scores
