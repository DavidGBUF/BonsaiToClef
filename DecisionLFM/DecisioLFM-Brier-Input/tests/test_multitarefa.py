"""Testes de alvos, separação de grupos e definições das métricas."""

import random
import unittest

from avaliar_tipos import metrics
from dados_tipos import encoded_record, group_key, make_record
from esquema import prepare_record
import test_modelo


class MultitaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_modelo.ModelTests.setUpClass()
        cls.tokenizer = test_modelo.ModelTests.tokenizer

    def test_targets_remain_outside_input_and_follow_choice_order(self):
        sample = {
            "state": {"text": "Cash withdrawal failed."},
            "task": "example",
            "type": "choice",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "What happened?",
                    "criteria": {"card": "Card payment", "cash": "Cash withdrawal"},
                }
            },
            "targets": {"q": "cash"},
        }
        for seed in range(5):
            record = encoded_record(self.tokenizer, sample, rng=random.Random(seed))
            self.assertEqual(record["option_ids"][record["target"]], "cash")
            self.assertNotIn("targets", self.tokenizer.decode(record["input_ids"]))

    def test_ordinal_order_and_binary_label_mapping(self):
        sample = make_record(
            "sst5",
            {"text": "Excellent movie.", "label": 4},
            None,
            1,
            "test",
            random.Random(42),
        )
        record = encoded_record(self.tokenizer, sample, rng=random.Random(3))
        self.assertEqual(record["option_ids"], ["0", "1", "2", "3", "4"])
        self.assertEqual(record["target"], 4)
        sample = make_record(
            "snli",
            {"premise": "A man runs.", "hypothesis": "Someone moves.", "label": 0},
            None,
            2,
            "test",
            random.Random(42),
        )
        _, questions = prepare_record(sample)
        self.assertEqual(next(iter(sample["targets"].values())), "true")
        self.assertEqual(next(iter(questions.values()))["type"], "noul")

    def test_groups_ignore_surface_spacing_and_snli_hypothesis(self):
        self.assertEqual(
            group_key("ag_news", {"text": " SAME  Document "}),
            group_key("ag_news", {"text": "same document"}),
        )
        self.assertEqual(
            group_key("snli", {"premise": "A man runs.", "hypothesis": "A"}),
            group_key("snli", {"premise": "A man runs.", "hypothesis": "B"}),
        )

    def test_metrics_brier_scale_failures_and_ordinal_expectation(self):
        base = {
            "task": "x",
            "type": "noul",
            "option_ids": ["true", "false"],
            "target": "true",
            "prediction": "true",
            "error": None,
            "probabilities": {"true": 1.0, "false": 0.0},
            "latency_ms": 10.0,
            "peak_allocated_vram_gib": 4.0,
        }
        self.assertEqual(metrics([base])["brier_multiclass"], 0)
        uniform = {**base, "probabilities": {"true": 0.5, "false": 0.5}}
        self.assertEqual(metrics([uniform])["brier_multiclass"], 0.5)
        wrong = {
            **base,
            "prediction": "false",
            "probabilities": {"true": 0.0, "false": 1.0},
        }
        self.assertEqual(metrics([wrong])["brier_multiclass"], 2)
        failed = {**base, "prediction": None, "probabilities": None, "error": "failure"}
        result = metrics([base, failed])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["failed_records"], 1)
        self.assertEqual(result["probability_metrics_records"], 1)
        score = {
            **base,
            "type": "score",
            "option_ids": ["0", "1", "2"],
            "target": "2",
            "prediction": "2",
            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
            "score": 1.6,
        }
        self.assertAlmostEqual(metrics([score])["score_mae"], 0.4)


if __name__ == "__main__":
    unittest.main()
