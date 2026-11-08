"""Contrato JSON, tipos e inferência de várias perguntas sem carregar o backbone."""

import json
import unittest
from unittest.mock import patch

from torch import nn

from esquema import format_answer, prepare_record
from modelo import DecisionLFM


QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What happened?",
        "criteria": {
            "card": "Card payment declined",
            "cash": "Cash withdrawal problem",
            "other": "Other banking issue",
        },
    },
    "urgent": {"type": "noul"},
    "severity": {"type": "score", "criteria": ["Low", "Medium", "High", "Critical"]},
}


class SchemaTests(unittest.TestCase):
    def test_json_states_and_question_ids(self):
        for state in [None, False, 42, ["á", 1], {"message": "Cartão", "attempts": 3}]:
            text, questions = prepare_record({"state": state, "questions": QUESTIONS})
            self.assertEqual(json.loads(text), state)
            self.assertEqual(list(questions), list(QUESTIONS))
            self.assertEqual(questions["urgent"]["instructions"], "urgent")
            self.assertEqual(
                list(questions["severity"]["options"]), ["0", "1", "2", "3"]
            )
        text, _ = prepare_record({"state": "Keep this text.", "questions": QUESTIONS})
        self.assertEqual(text, "Keep this text.")

    def test_typed_answers(self):
        _, questions = prepare_record({"state": "", "questions": QUESTIONS})
        choice = format_answer(
            questions["intent"], {"card": 0.8, "cash": 0.1, "other": 0.1}
        )
        self.assertEqual(choice["choice"], "card")
        self.assertEqual(choice["confidence"], 0.8)
        self.assertEqual(
            format_answer(questions["urgent"], {"true": 0.7, "false": 0.3}),
            {"type": "noul", "noul": 0.7},
        )
        score = format_answer(
            questions["severity"], {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}
        )
        self.assertAlmostEqual(score["score"], 2.0)
        self.assertEqual(score["legend"]["3"], "Critical")

    def test_boolean_overrides_and_invalid_records(self):
        _, questions = prepare_record(
            {
                "state": "",
                "questions": {"q": {"type": "noul", "criteria": {"true": "Urgent"}}},
            }
        )
        self.assertEqual(questions["q"]["options"]["true"], "Urgent")
        self.assertIn("false", questions["q"]["options"])
        for record in [
            None,
            {},
            {"state": "", "questions": {}},
            {"state": float("nan"), "questions": QUESTIONS},
        ]:
            with self.subTest(record=record), self.assertRaises(ValueError):
                prepare_record(record)
        for question in [
            {"type": "unknown"},
            {"type": "choice", "criteria": {"a": "A"}},
            {"type": "choice", "criteria": {"a": "A", "b": ""}},
            {"type": "score", "criteria": {"0": "Low", "1": "High"}},
            {"type": "noul", "criteria": {"yes": "Yes"}},
            {"type": "noul", "instructions": False},
        ]:
            with self.subTest(question=question), self.assertRaises(ValueError):
                prepare_record({"state": "", "questions": {"q": question}})


class JsonModelTests(unittest.TestCase):
    def test_multiple_questions_use_existing_decider(self):
        model = DecisionLFM.__new__(DecisionLFM)
        nn.Module.__init__(model)
        state = {"message": "My card was declined.", "attempts": 3}

        def decide(text, instructions, options, temperature=1.0):
            self.assertEqual(json.loads(text), state)
            total = sum(range(1, len(options) + 1))
            probabilities = {key: i / total for i, key in enumerate(options, 1)}
            return {"probabilities": probabilities, "input_tokens": 10}

        with patch.object(model, "decidir", side_effect=decide) as predict:
            result = model.responder({"state": state, "questions": QUESTIONS})
            self.assertEqual(predict.call_count, 3)
            self.assertEqual(predict.call_args_list[1].args[1], "urgent")
        self.assertEqual(list(result["answers"]), list(QUESTIONS))
        self.assertEqual(result["usage"], {"input_tokens": 30, "output_tokens": 0})
        self.assertEqual(result["answers"]["intent"]["choice"], "other")
        self.assertAlmostEqual(result["answers"]["urgent"]["noul"], 1 / 3)
        self.assertAlmostEqual(result["answers"]["severity"]["score"], 2.0)
        with patch.object(model, "decidir") as predict, self.assertRaises(ValueError):
            model.responder(
                {"state": state, "questions": {**QUESTIONS, "bad": {"type": "bad"}}}
            )
        predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()

