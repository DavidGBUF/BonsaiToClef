"""Testes pequenos de alinhamento, alternativas variáveis e máscara da cabeça."""

import os
import unittest

import torch

from modelo import DecisionHead, MODEL, REVISION, collate, encode


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            os.environ.get("DECISIONLFM_TEST_TOKENIZER", MODEL),
            revision=REVISION,
            code_revision=REVISION,
            token=False,
            trust_remote_code=True,
        )

    def test_unicode_offsets_and_ids_outside_prompt(self):
        options = {
            "hidden_id_1": "Cartão não chegou",
            "hidden_id_2": "Saque em dinheiro",
        }
        record = encode(
            self.tokenizer, "Meu cartão não chegou.", "Qual é a intenção?", options
        )
        for description, positions in zip(options.values(), record["option_positions"]):
            text = self.tokenizer.decode([record["input_ids"][i] for i in positions])
            self.assertIn(description, text)
            self.assertLess(max(positions), len(record["input_ids"]) - 1)
        decoded = self.tokenizer.decode(record["input_ids"])
        self.assertNotIn("hidden_id_1", decoded)
        self.assertEqual(record["option_ids"], list(options))

    def test_dynamic_choices_padding_and_pooling(self):
        a = encode(self.tokenizer, "Hi", "Choose", {"a": "Alpha", "b": "Beta"})
        b = encode(
            self.tokenizer,
            "A longer state.",
            "Choose",
            {"x": "First", "y": "Second", "z": "Third"},
        )
        batch = collate([a, b], self.tokenizer.pad_token_id)
        self.assertEqual(
            batch["option_mask"].tolist(), [[True, True, False], [True, True, True]]
        )
        self.assertTrue(
            torch.allclose(
                batch["option_weights"].sum(-1), batch["option_mask"].float()
            )
        )
        self.assertTrue(
            torch.all(
                batch["option_weights"] * (1 - batch["attention_mask"][:, None]) == 0
            )
        )
        self.assertEqual(
            batch["attention_mask"].sum(-1).tolist(),
            [len(a["input_ids"]), len(b["input_ids"])],
        )

    def test_limit_rejected_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "limite"):
            encode(self.tokenizer, "Hi", "Choose", {"a": "A", "b": "B"}, max_length=2)
        with self.assertRaises(ValueError):
            encode(self.tokenizer, "Hi", "Choose", {"a": "A"})

    def test_shared_scorer_permutation_and_mask(self):
        torch.manual_seed(7)
        head = DecisionHead(16, 8)
        q, options = torch.randn(2, 16), torch.randn(2, 4, 16)
        mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
        logits = head(q, options, mask)
        order = [3, 1, 0, 2]
        self.assertTrue(
            torch.allclose(head(q, options[:, order], mask[:, order]), logits[:, order])
        )
        p = logits.softmax(-1)
        self.assertEqual(p[0, 2:].sum().item(), 0)
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones(2)))
        (-p[:, 0].log().mean()).backward()
        self.assertTrue(torch.isfinite(head.query.weight.grad).all())
        self.assertGreater(head.query.weight.grad.norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
