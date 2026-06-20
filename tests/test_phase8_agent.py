"""Phase 08 — agent mode: the AgentSession tool layer + MCP scaffolding.

Drives the three compute axes (Memory / Movement / Truth) through
``backend.agent.AgentSession`` with a scripted scorer, and checks the optional
MCP server wiring degrades gracefully when the ``mcp`` package is absent.
"""

import asyncio
import unittest

from data.schema import Chunk
from inference.scorer import PrefixState, ScoreRequest, ScoreResult, ScorerClient

from backend.agent import AgentSession


class ScriptedScorer(ScorerClient):
    """Deterministic scorer: 'retry'/'network' match the demo retry chunks; any
    'python' predicate (the beam's facet-narrowed clauses) scores moderately."""

    def __init__(self) -> None:
        self.calls = 0

    async def warm(self, corpus_id: str, chunks: list[Chunk]) -> PrefixState:
        return PrefixState(corpus_id, len(chunks), True, self.model_id())

    async def score_batch(self, items: list[ScoreRequest], *, tier: int = 1) -> list[ScoreResult]:
        self.calls += 1
        results = []
        for item in items:
            text = item.chunk_text.lower()
            predicate = item.predicate.lower()
            score = 0.2
            if "retry" in predicate and ("retry" in text or "retries" in text):
                score = 0.9
            if "network" in predicate and ("network" in text or "http" in text):
                score = max(score, 0.85)
            if "python" in predicate:
                score = max(score, 0.7)
            results.append(
                ScoreResult(chunk_id=item.chunk_id, score=score, p_yes=score, p_no=1.0 - score, tier=tier)
            )
        return results

    async def health(self) -> dict:
        return {"ready": True, "backend": "scripted"}

    def model_id(self) -> str:
        return "scripted-scorer"


def session() -> AgentSession:
    agent = AgentSession(ScriptedScorer())
    asyncio.run(agent.ingest("demo"))
    return agent


class AgentAxis1MemoryTests(unittest.TestCase):
    def test_full_budget_scores_whole_corpus(self):
        agent = session()
        out = asyncio.run(agent.query("retry", compute_budget=1.0))
        self.assertEqual(out["corpus_total"], out["corpus_scored"])
        self.assertGreater(out["matched"], 0)

    def test_partial_budget_scores_fewer_chunks(self):
        agent = session()
        out = asyncio.run(agent.query("retry", compute_budget=0.5))
        self.assertEqual(out["corpus_total"], 7)
        self.assertLess(out["corpus_scored"], out["corpus_total"])
        self.assertGreaterEqual(out["corpus_scored"], 1)

    def test_results_is_a_pure_cache_read(self):
        agent = session()
        asyncio.run(agent.query("retry"))
        before = agent.scorer.calls
        view = agent.results(threshold=0.5)
        self.assertEqual(agent.scorer.calls, before)  # no inference
        self.assertGreater(view["matched"], 0)


class AgentAxis2MovementTests(unittest.TestCase):
    def test_threshold_mode_auto_sets_cutoff(self):
        agent = session()
        asyncio.run(agent.query("retry"))
        out = agent.select(mode="threshold", precision_target=0.85)
        self.assertEqual(out["mode"], "threshold")
        self.assertGreater(out["selected_count"], 0)

    def test_smart_mode_covers_facets_and_beats_greedy_floor(self):
        agent = session()
        asyncio.run(agent.query("retry"))
        before = agent.scorer.calls
        out = agent.select(mode="smart", precision_target=0.5, movement_budget=3, beam_width=4)
        self.assertEqual(agent.scorer.calls, before)  # zero inference
        self.assertEqual(out["mode"], "smart")
        self.assertLessEqual(out["selected_count"], 3)
        self.assertGreater(len(out["covered_facets"]), 0)
        self.assertGreaterEqual(out["objective"], out["greedy_objective"])


class AgentAxis3TruthTests(unittest.TestCase):
    def test_refine_runs_beam_and_applies_winner(self):
        agent = session()
        asyncio.run(agent.query("retry"))
        out = asyncio.run(agent.refine("only python", beam_width=4))
        self.assertTrue(out["chosen"].startswith("only python"))
        self.assertGreaterEqual(len(out["candidates"]), 1)
        self.assertLessEqual(len(out["candidates"]), 4)
        self.assertEqual(sum(1 for c in out["candidates"] if c["chosen"]), 1)
        self.assertIn("matched", out)

    def test_beam_width_one_explores_single_candidate(self):
        agent = session()
        asyncio.run(agent.query("retry"))
        out = asyncio.run(agent.refine("only python", beam_width=1))
        self.assertEqual(len(out["candidates"]), 1)
        self.assertEqual(out["candidates"][0]["text"], "only python")

    def test_refine_before_query_raises(self):
        agent = session()
        with self.assertRaises(ValueError):
            asyncio.run(agent.refine("only python"))


class McpServerTests(unittest.TestCase):
    def test_mcp_available_returns_bool(self):
        from backend.mcp_server import mcp_available

        self.assertIsInstance(mcp_available(), bool)

    def test_build_server_matches_availability(self):
        from backend import mcp_server

        if mcp_server.mcp_available():
            server = mcp_server.build_server(AgentSession(ScriptedScorer()))
            self.assertIsNotNone(server)
        else:
            with self.assertRaises(RuntimeError):
                mcp_server.build_server(AgentSession(ScriptedScorer()))


if __name__ == "__main__":
    unittest.main()
