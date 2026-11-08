"""Verifica escala, gradientes, padding e acumulação da loss usada no treino."""

import math
import unittest

import torch
from torch.nn import functional as F

from objetivo import decision_loss


class LossTests(unittest.TestCase):
    def test_zero_weight_preserves_cross_entropy_and_gradients(self):
        logits = torch.tensor(
            [[1.0, -2.0, -torch.inf], [0.2, 0.5, 1.0]], requires_grad=True
        )
        target = torch.tensor([0, 2])
        for reduction in ["none", "mean", "sum"]:
            expected = F.cross_entropy(logits, target, reduction=reduction)
            actual = decision_loss(logits, target, 0, reduction)
            self.assertTrue(torch.equal(actual, expected))
        a = torch.autograd.grad(decision_loss(logits, target, 0), logits)[0]
        b = torch.autograd.grad(F.cross_entropy(logits, target), logits)[0]
        self.assertTrue(torch.equal(a, b))

    def test_brier_scale_and_weight(self):
        logits = torch.zeros(1, 2)
        target = torch.tensor([0])
        self.assertAlmostEqual(decision_loss(logits, target).item(), math.log(2) + 0.5)
        self.assertAlmostEqual(
            decision_loss(logits, target, 0.25).item(), math.log(2) + 0.125
        )
        wrong = torch.tensor([[-100.0, 100.0]])
        self.assertAlmostEqual(
            (decision_loss(wrong, target) - decision_loss(wrong, target, 0)).item(), 2.0
        )

    def test_mixed_choice_noul_score_ignore_padding_and_backpropagate(self):
        torch.manual_seed(7)
        sizes = [77, 2, 5]
        logits = torch.randn(3, 77)
        for index, size in enumerate(sizes):
            logits[index, size:] = -torch.inf
        logits.requires_grad_()
        target = torch.tensor([23, 1, 4])
        values = decision_loss(logits, target, reduction="none")
        for index, size in enumerate(sizes):
            torch.testing.assert_close(
                values[index],
                decision_loss(
                    logits[index : index + 1, :size], target[index : index + 1]
                ),
            )
        values.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        for index, size in enumerate(sizes):
            self.assertGreater(logits.grad[index, :size].abs().sum().item(), 0)
            self.assertEqual(logits.grad[index, size:].abs().sum().item(), 0)

    def test_unequal_microbatches_preserve_full_batch_gradient(self):
        torch.manual_seed(42)
        inputs, weight = torch.randn(5, 4), torch.randn(4, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 1, 0])
        full = torch.autograd.grad(decision_loss(inputs @ weight, targets), weight)[0]
        accumulated = torch.zeros_like(weight)
        for start, end in [(0, 2), (2, 4), (4, 5)]:
            loss = decision_loss(
                inputs[start:end] @ weight, targets[start:end], reduction="sum"
            )
            accumulated += torch.autograd.grad(loss, weight)[0]
        torch.testing.assert_close(accumulated / len(inputs), full)

    def test_invalid_parameters(self):
        for value in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                decision_loss(torch.zeros(1, 2), torch.tensor([0]), value)
        with self.assertRaises(ValueError):
            decision_loss(torch.zeros(1, 2), torch.tensor([0]), reduction="bad")

    def test_metal_memory_is_distinct_from_dedicated_vram(self):
        from avaliar_tipos import metrics

        row = {
            "task": "binary",
            "type": "noul",
            "option_ids": ["true", "false"],
            "target": "true",
            "prediction": "true",
            "probabilities": {"true": 0.8, "false": 0.2},
            "error": None,
            "latency_ms": 20.0,
            "peak_allocated_vram_gib": None,
            "mps_allocated_gib": 4.0,
        }
        result = metrics([row])
        self.assertIsNone(result["peak_allocated_vram_gib"])
        self.assertEqual(result["max_observed_mps_allocated_gib"], 4.0)


if __name__ == "__main__":
    unittest.main()

