"""Phase 06 freeze guard: the three rehearsals pass, the freeze manifest is
reproducible from saved artifacts, and the docs agree on the frozen numbers."""
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class FreezeTests(unittest.TestCase):
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

    def test_three_rehearsals_pass(self):
        from eval.rehearsal import run_rehearsals

        report = run_rehearsals()
        self.assertTrue(report.live, report.notes)
        self.assertTrue(report.fallback, report.notes)
        self.assertTrue(report.timed, report.notes)

    def test_timed_budget_under_90s(self):
        from eval.rehearsal import BEAT_BUDGET_S

        self.assertLessEqual(sum(s for _, s in BEAT_BUDGET_S), 90)

    def test_freeze_manifest_matches_saved_artifacts(self):
        import json

        from eval.rehearsal import ARTIFACT_DIR, build_freeze_manifest

        manifest = build_freeze_manifest()
        gate = manifest["frozen_metrics"]["quality_gate"]
        # The manifest must echo the measured real gate, not a placeholder.
        self.assertEqual(gate["label"], "measured")
        self.assertGreaterEqual(gate["f1"], 0.7)

        # And it must equal what the saved artifact actually says (reproducible).
        artifact = json.loads((ARTIFACT_DIR / "phase04_quality_gate.json").read_text())
        self.assertEqual(gate["f1"], artifact["quality"]["f1"])
        self.assertEqual(gate["recommended_threshold"], artifact["recommended_threshold"])

    def test_docs_agree_on_frozen_claims(self):
        slide = (REPO / "eval" / "SLIDE.md").read_text()
        demo = (REPO / "DEMO.md").read_text()
        bench = (REPO / "eval" / "bench.py").read_text()
        freeze = (REPO / "FREEZE.md").read_text()

        # Gate is presented as measured (real), with the threshold-calibration caveat.
        self.assertIn("0.94", slide)
        self.assertIn("0.016", slide)
        # The real-backend threshold caveat is surfaced to the operator.
        self.assertIn("0.016", demo)
        # The enforced gate threshold matches the docs.
        self.assertIn("MIN_F1 = 0.7", bench)
        self.assertIn("0.7", freeze)
        # MFU honesty is preserved everywhere it appears.
        self.assertIn("under-saturated", slide.lower())


if __name__ == "__main__":
    unittest.main()
