import unittest


class Phase2RefineReplayTests(unittest.TestCase):
    def test_cumulative_curves_compare_scoped_full_suffix_and_rag(self):
        from eval.refine_replay import RefineTraceTurn, cumulative_curves

        turns = [
            RefineTraceTurn(candidate_count=4, chunks_scored=2, survivor_count=3),
            RefineTraceTurn(candidate_count=6, chunks_scored=3, survivor_count=2),
        ]

        curves = cumulative_curves(turns, n_chunks_total=10, rag_reindex_turns={2})

        self.assertEqual(curves["scoped"], [2, 5])
        self.assertEqual(curves["full"], [10, 20])
        self.assertEqual(curves["suffix"], [4, 10])
        self.assertEqual(curves["rag"], [10, 30])


if __name__ == "__main__":
    unittest.main()
