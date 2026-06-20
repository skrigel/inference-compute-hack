"""Phase 03 cut-line regression guard.

Asserts the irreducible demo loop is green and that the load-bearing invariants
hold: a cold query scans the whole corpus, click-NOT and threshold drag are
zero-inference, the AND refine scores only a scoped subset, and a fresh file is
queryable immediately.
"""
import unittest


class CutLineTests(unittest.TestCase):
    def setUp(self):
        from eval.cut_line import run_cut_line

        self.result = run_cut_line()

    def tearDown(self):
        import itertools

        import backend.main as main
        from backend.cache import ScoreCache
        from backend.state import BackendState
        from inference.mock_scorer import MockScorer

        main.state = BackendState()
        main.cache = ScoreCache()
        main.scorer = MockScorer()
        main._clause_seq = itertools.count(1)

    def _step(self, name: str):
        return next(step for step in self.result.steps if step.name == name)

    def test_loop_is_green(self):
        self.assertTrue(self.result.green, f"cut-line failures: {self.result.failures}")

    def test_cold_query_scans_whole_corpus(self):
        query = self._step("query")
        self.assertEqual(query.chunks_scored, self.result.n_chunks)
        self.assertGreaterEqual(query.matched, 2)

    def test_click_not_is_zero_inference(self):
        self.assertEqual(self._step("click-NOT").chunks_scored, 0)

    def test_and_refine_scores_only_a_scoped_subset(self):
        refine = self._step("AND refine")
        # Rigorous scoping proof: the refine scores EXACTLY the current survivor
        # set (its candidate_count == the prior step's matched count), and that
        # set is a strict subset of the corpus.
        self.assertGreater(refine.chunks_scored, 0)
        self.assertEqual(refine.chunks_scored, refine.candidate_count)
        self.assertEqual(refine.candidate_count, self._step("click-NOT").matched)
        self.assertLess(refine.chunks_scored, self.result.n_chunks)
        self.assertGreaterEqual(refine.matched, 1)

    def test_threshold_drag_is_zero_inference(self):
        self.assertEqual(self._step("threshold drag").chunks_scored, 0)

    def test_fresh_file_is_queryable_with_zero_derived_bytes(self):
        self.assertGreaterEqual(self._step("fresh-file").matched, 1)
        self.assertEqual(self.result.fresh_vs_rag["ours"]["derived_bytes_written"], 0)
        self.assertTrue(self.result.fresh_vs_rag["rag"]["reindex_required"])

    def test_area_under_loop_scoped_beats_full(self):
        aul = self.result.area_under_loop
        self.assertLess(aul["scoped_total"], aul["full_total"])


if __name__ == "__main__":
    unittest.main()
