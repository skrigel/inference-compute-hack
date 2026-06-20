import unittest


class EvalConfigTests(unittest.TestCase):
    def test_regimes_match_eval_plan_ladder(self):
        from eval.config import REGIMES, ComputeRegime

        self.assertEqual(
            list(REGIMES.keys()),
            ["B0_baseline", "B1_warm", "B2_scoped", "B3_cached"],
        )
        self.assertEqual(REGIMES["B0_baseline"], ComputeRegime.baseline())
        self.assertEqual(REGIMES["B1_warm"], ComputeRegime.warm())
        self.assertEqual(REGIMES["B2_scoped"], ComputeRegime.scoped())
        self.assertEqual(REGIMES["B3_cached"], ComputeRegime.cached())

    def test_throughput_ladder_separates_fp8_from_four_bit_capacity(self):
        from eval.config import CAPACITY_LADDER, THROUGHPUT_LADDER

        self.assertIn("A3_fp8_compute", THROUGHPUT_LADDER)
        self.assertNotIn("A3_4bit", THROUGHPUT_LADDER)
        self.assertIn("C1_4bit_weights_kv", CAPACITY_LADDER)


if __name__ == "__main__":
    unittest.main()
