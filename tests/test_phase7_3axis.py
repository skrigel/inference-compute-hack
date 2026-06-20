"""Phase 07 — backend tests for the Infinite-Compute 3-Axis extensions.

Axis 1 (Memory): compute_budget scopes the corpus scored per query.
Axis 2 (Movement): /select auto-threshold (Mode A) + smart-select (Mode B).
Axis 3 (Truth): beam_width refine explores candidates and objective-selects.
"""

import asyncio
import itertools
import json
import unittest

from data.schema import Chunk, ChunkMeta, chunk_id_of
from inference.scorer import PrefixState, ScoreRequest, ScoreResult, ScorerClient


def sse_events(response) -> list[dict]:
    events = []
    for frame in response.text.strip().split("\n\n"):
        if not frame:
            continue
        assert frame.startswith("data: "), frame
        events.append(json.loads(frame.removeprefix("data: ")))
    return events


class ScriptedScorer(ScorerClient):
    """Deterministic scorer: 'retry' predicates score retry/network chunks high."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.tiers: list[int] = []

    async def warm(self, corpus_id: str, chunks: list[Chunk]) -> PrefixState:
        return PrefixState(corpus_id, len(chunks), True, self.model_id())

    async def score_batch(self, items: list[ScoreRequest], *, tier: int = 1) -> list[ScoreResult]:
        self.batch_sizes.append(len(items))
        self.tiers.append(tier)
        results = []
        for item in items:
            text = item.chunk_text.lower()
            predicate = item.predicate.lower()
            score = 0.12
            if "retry" in predicate and ("retry" in text or "retries" in text):
                score = 0.9
            if "network" in predicate and ("network" in text or "http" in text):
                score = 0.86
            results.append(
                ScoreResult(
                    chunk_id=item.chunk_id,
                    score=score,
                    p_yes=score,
                    p_no=1.0 - score,
                    tier=tier,
                )
            )
        return results

    async def health(self) -> dict:
        return {"ready": True, "backend": "scripted"}

    def model_id(self) -> str:
        return "scripted-scorer"


def reset_backend(scorer: ScorerClient | None = None):
    import backend.main as main
    from backend.cache import ScoreCache
    from backend.state import BackendState

    main.state = BackendState()
    main.cache = ScoreCache()
    main.scorer = scorer or ScriptedScorer()
    main._clause_seq = itertools.count(1)
    return main


class Axis1ComputeBudgetTests(unittest.TestCase):
    def tearDown(self):
        reset_backend()

    def _run(self, budget: float, batch_size: int = 2):
        from backend.cache import ScoreCache
        from backend.streaming import query_stream

        cache = ScoreCache()
        scorer = ScriptedScorer()
        chunks = [
            Chunk(
                chunk_id=chunk_id_of(f"d{i}", 0, f"retry without backoff {i}"),
                doc_id=f"d{i}",
                type="code",
                title=f"d{i}",
                text=f"retry without backoff {i}",
                meta=ChunkMeta("python", 2024, f"d{i}.py", "python", "demo", "synthetic"),
            )
            for i in range(10)
        ]

        async def go():
            events = []
            async for event in query_stream(
                scorer,
                chunks,
                "retry",
                clause_id="q1",
                threshold=0.5,
                cache=cache,
                batch_size=batch_size,
                compute_budget=budget,
            ):
                events.append(event)
            return events

        return asyncio.run(go()), scorer

    def test_full_budget_scores_whole_corpus(self):
        events, scorer = self._run(1.0)
        done = events[-1]
        self.assertEqual(done.corpus_total, 10)
        self.assertEqual(done.corpus_scored, 10)
        self.assertEqual(sum(scorer.batch_sizes), 10)

    def test_half_budget_scores_half_the_corpus(self):
        events, scorer = self._run(0.5)
        done = events[-1]
        self.assertEqual(done.corpus_total, 10)
        self.assertEqual(done.corpus_scored, 5)
        self.assertEqual(sum(scorer.batch_sizes), 5)

    def test_small_budget_scores_at_least_one_chunk(self):
        events, scorer = self._run(0.01)
        done = events[-1]
        self.assertEqual(done.corpus_total, 10)
        self.assertEqual(done.corpus_scored, 1)

    def test_query_endpoint_passes_budget_through(self):
        from fastapi.testclient import TestClient

        main = reset_backend(ScriptedScorer())
        client = TestClient(main.app)
        client.post("/ingest", json={"corpus_id": "demo"})
        events = sse_events(
            client.post("/query", json={"predicate": "retry", "threshold": 0.5, "compute_budget": 0.5})
        )
        done = next(event for event in events if event["type"] == "done")
        self.assertLess(done["corpus_scored"], done["corpus_total"])
        self.assertEqual(done["compute_budget"], 0.5)


class Axis2SelectTests(unittest.TestCase):
    def tearDown(self):
        reset_backend()

    def test_auto_threshold_hits_precision_target(self):
        from backend.select import auto_threshold

        scores = [0.95, 0.9, 0.8, 0.4, 0.1]
        threshold, count = auto_threshold(scores, 0.85)
        # mean of {0.95, 0.9, 0.8} = 0.883 >= 0.85; adding 0.4 would drop below.
        self.assertEqual(count, 3)
        self.assertAlmostEqual(threshold, 0.8)

    def test_auto_threshold_empty_and_unreachable(self):
        from backend.select import auto_threshold

        self.assertEqual(auto_threshold([], 0.85), (1.0, 0))
        threshold, count = auto_threshold([0.4, 0.3], 0.85)
        self.assertEqual(count, 0)

    def test_smart_select_beam_beats_or_matches_greedy(self):
        from backend.select import max_coverage_select

        # Two high-score chunks share facet A; a lower chunk uniquely covers B.
        items = [
            ("a", {"f:A"}, 0.95),
            ("b", {"f:A"}, 0.90),
            ("c", {"f:B"}, 0.60),
        ]
        selected, covered, objective, greedy = max_coverage_select(items, movement_budget=2, beam_width=4)
        self.assertEqual(len(selected), 2)
        # Best 2-subset for coverage is {A-chunk, B-chunk}: 2 facets covered.
        self.assertEqual(len(covered), 2)
        self.assertGreaterEqual(objective, greedy)

    def test_select_endpoint_threshold_mode_is_zero_inference(self):
        from fastapi.testclient import TestClient

        scorer = ScriptedScorer()
        main = reset_backend(scorer)
        client = TestClient(main.app)
        client.post("/ingest", json={"corpus_id": "demo"})
        client.post("/query", json={"predicate": "retry", "threshold": 0.5})
        scorer.batch_sizes.clear()

        response = client.post("/select", json={"mode": "threshold", "precision_target": 0.8})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "threshold")
        self.assertEqual(scorer.batch_sizes, [])  # pure cache read
        self.assertEqual(body["selected_count"], len(body["selected_ids"]))

    def test_select_endpoint_smart_mode_returns_coverage(self):
        from fastapi.testclient import TestClient

        scorer = ScriptedScorer()
        main = reset_backend(scorer)
        client = TestClient(main.app)
        client.post("/ingest", json={"corpus_id": "demo"})
        client.post("/query", json={"predicate": "retry", "threshold": 0.5})
        scorer.batch_sizes.clear()

        response = client.post(
            "/select",
            json={"mode": "smart", "precision_target": 0.5, "movement_budget": 3, "beam_width": 4},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "smart")
        self.assertEqual(scorer.batch_sizes, [])
        self.assertLessEqual(len(body["selected_ids"]), 3)
        self.assertGreaterEqual(body["objective"], body["greedy_objective"])


class Axis3BeamRefineTests(unittest.TestCase):
    def tearDown(self):
        reset_backend()

    def test_beam_width_one_matches_classic_refine(self):
        from fastapi.testclient import TestClient

        main = reset_backend(ScriptedScorer())
        client = TestClient(main.app)
        client.post("/ingest", json={"corpus_id": "demo"})
        client.post("/query", json={"predicate": "retry", "threshold": 0.5})

        events = sse_events(client.post("/refine", json={"utterance": "only networking layer", "beam_width": 1}))
        self.assertEqual([event["type"] for event in events], ["chip", "diff", "aggregate", "done"])

    def test_beam_width_emits_beam_event_and_selects_candidate(self):
        from fastapi.testclient import TestClient

        main = reset_backend(ScriptedScorer())
        client = TestClient(main.app)
        client.post("/ingest", json={"corpus_id": "demo"})
        client.post("/query", json={"predicate": "retry", "threshold": 0.5})

        events = sse_events(client.post("/refine", json={"utterance": "networking layer", "beam_width": 4}))
        self.assertEqual(events[0]["type"], "beam")
        beam = events[0]
        self.assertGreater(len(beam["candidates"]), 1)
        chosen = [c for c in beam["candidates"] if c["chosen"]]
        self.assertEqual(len(chosen), 1)
        self.assertEqual(beam["candidates"][beam["chosen_index"]]["chosen"], True)
        # The chip/diff/aggregate/done still follow the beam event.
        self.assertEqual([event["type"] for event in events[1:]], ["chip", "diff", "aggregate", "done"])


if __name__ == "__main__":
    unittest.main()
