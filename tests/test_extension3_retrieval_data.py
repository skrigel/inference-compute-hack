import importlib.util
import json
import unittest
from pathlib import Path


def _load_retrieval_data_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "environments"
        / "extension3_agent_loop"
        / "extension3_agent_loop"
        / "retrieval_data.py"
    )
    spec = importlib.util.spec_from_file_location("extension3_retrieval_data_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Extension3RetrievalDataTests(unittest.TestCase):
    def test_large_retrieval_rows_are_serializable_and_diverse(self):
        data = _load_retrieval_data_module()

        train_rows = data.build_retrieval_rows(split="train", max_examples=-1, include_hard=True)
        eval_rows = data.build_retrieval_rows(split="eval", max_examples=-1, include_hard=True)

        self.assertGreaterEqual(len(train_rows), 1000)
        self.assertGreaterEqual(len(eval_rows), 200)
        self.assertGreaterEqual(len({row["domain"] for row in train_rows}), 10)
        self.assertTrue({row["task_id"] for row in train_rows}.isdisjoint({row["task_id"] for row in eval_rows}))

        answer = json.loads(train_rows[0]["answer"])
        self.assertIsInstance(train_rows[0]["answer"], str)
        self.assertGreaterEqual(len(answer["positive_ids"]), 2)
        self.assertGreaterEqual(len(answer["hard_negative_ids"]), 2)
        self.assertGreater(len(train_rows[0]["prompt"][1]["content"].split()), 450)

    def test_reward_prefers_precise_evidence_over_select_everything(self):
        data = _load_retrieval_data_module()
        row = data.build_retrieval_rows(split="eval", max_examples=1, include_hard=True)[0]
        answer = json.loads(row["answer"])

        good_completion = json.dumps(
            {
                "refined_query": answer["target_query"],
                "evidence_ids": answer["positive_ids"][:3],
                "exclude_terms": answer["exclude_terms"][:2],
                "stop": True,
            }
        )
        bad_completion = json.dumps(
            {
                "refined_query": f"{answer['initial_query']} all documents entire corpus",
                "evidence_ids": answer["positive_ids"][:1] + answer["hard_negative_ids"][:2],
                "exclude_terms": [],
                "stop": True,
            }
        )

        good = data.score_completion(good_completion, row["answer"])
        bad = data.score_completion(bad_completion, row["answer"])

        self.assertGreaterEqual(good["reward"], 0.85)
        self.assertEqual(good["hard_negative_rejection"], 1.0)
        self.assertGreater(good["evidence_id_recall"], bad["evidence_id_recall"])
        self.assertLess(bad["anti_select_all"], 0.5)
        self.assertGreater(good["reward"], bad["reward"] + 0.35)


if __name__ == "__main__":
    unittest.main()
