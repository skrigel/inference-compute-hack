from __future__ import annotations

from dataclasses import dataclass, field

from inference.scorer import ScoreResult


@dataclass
class ScoreCache:
    """Per-(chunk_id, clause_id) score cache — the B2/B3 latency win.

    Thresholding and (in Phase 2) chip removal are pure reads over this map, so
    they cost zero inference. ``clause_id`` is minted by the backend; the scorer
    never sees it. In Phase 1 there is a single base clause per active query.
    """

    _scores: dict[tuple[str, str], ScoreResult] = field(default_factory=dict)
    _hits: int = 0
    _misses: int = 0

    def clear(self) -> None:
        self._scores.clear()
        self._hits = 0
        self._misses = 0

    def get(self, chunk_id: str, clause_id: str) -> ScoreResult | None:
        result = self._scores.get((chunk_id, clause_id))
        if result is None:
            self._misses += 1
        else:
            self._hits += 1
        return result

    def peek(self, chunk_id: str, clause_id: str) -> ScoreResult | None:
        return self._scores.get((chunk_id, clause_id))

    def put(self, chunk_id: str, clause_id: str, result: ScoreResult) -> None:
        self._scores[(chunk_id, clause_id)] = result

    def put_many(self, clause_id: str, results: list[ScoreResult]) -> None:
        for result in results:
            self._scores[(result.chunk_id, clause_id)] = result

    def scores_for_clause(self, clause_id: str) -> dict[str, ScoreResult]:
        return {
            chunk_id: result
            for (chunk_id, cid), result in self._scores.items()
            if cid == clause_id
        }

    def missing(self, clause_id: str, candidate_ids: set[str]) -> set[str]:
        """Candidate chunks with no cached score — the only ones needing inference."""
        missing = set()
        for chunk_id in candidate_ids:
            if (chunk_id, clause_id) in self._scores:
                self._hits += 1
            else:
                self._misses += 1
                missing.add(chunk_id)
        return missing

    def evict_clause(self, clause_id: str) -> None:
        for key in [k for k in self._scores if k[1] == clause_id]:
            del self._scores[key]

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def stats(self) -> dict:
        clauses = {clause_id for _, clause_id in self._scores}
        return {
            "n_entries": len(self._scores),
            "n_clauses": len(clauses),
            "hit_rate": self.hit_rate(),
        }
