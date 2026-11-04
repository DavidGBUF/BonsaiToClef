"""Calibração deve reduzir confiança sem mudar classes nem misturar documentos."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from calibracao import (
    fit_by_type,
    fit_temperature,
    load_calibration,
    probabilities,
    validation_partition,
    weight_hash,
    choose_candidate,
    summarize_logits,
)
from objetivo import decision_loss
from treinar_tipos import validate


class CalibrationTests(unittest.TestCase):
    def test_score_constraint_keeps_t1_when_softening_worsens_mae(self):
        logits = torch.tensor([[8.0, 0.0]] * 10 + [[0.0, 8.0]] * 10)
        targets = torch.tensor([0] * 7 + [1] * 3 + [1] * 7 + [0] * 3)
        _, unconstrained = fit_temperature(logits, targets)
        temperature, protected = fit_temperature(logits, targets, protect_mae=True)
        self.assertGreater(unconstrained["mae_after"], unconstrained["mae_before"])
        self.assertEqual(temperature, 1.0)
        self.assertLessEqual(protected["mae_after"], protected["mae_before"] + 1e-12)
        self.assertLessEqual(protected["after"], protected["before"])

    def test_candidate_selection_prioritizes_accuracy_and_rejects_worse_mae(self):
        def candidate(name, acc, brier, mae):
            return {
                "name": name,
                "validation": {
                    "mean_type_accuracy": acc,
                    "mean_type_brier": brier,
                    "types": {"score": {"mae": mae}},
                },
            }

        baseline = candidate("baseline", 0.7, 0.4, 0.5)
        items = [
            baseline,
            candidate("higher-accuracy", 0.8, 0.39, 0.49),
            candidate("lower-brier", 0.79, 0.2, 0.49),
            candidate("worse-mae", 0.9, 0.1, 0.6),
        ]
        self.assertEqual(
            choose_candidate(items, baseline["validation"])["name"], "higher-accuracy"
        )
        self.assertEqual(
            choose_candidate([baseline, items[-1]], baseline["validation"])["name"],
            "baseline",
        )

    def test_validation_summary_preserves_argmax_but_reports_score_mae(self):
        rows = [
            {"type": kind, "logits": [3.0, 0.0], "target": 0}
            for kind in ["choice", "noul", "score"]
        ]
        raw, scaled = summarize_logits(rows), summarize_logits(rows, {"score": 2.0})
        self.assertEqual(raw["mean_type_accuracy"], scaled["mean_type_accuracy"])
        self.assertGreater(
            scaled["types"]["score"]["mae"], raw["types"]["score"]["mae"]
        )

    def test_ordered_experiments_change_one_hyperparameter_per_parent_edge(self):
        from comparar_ajustes import CONFIGURATIONS

        configs = {c["name"]: c for c in CONFIGURATIONS}
        for c in CONFIGURATIONS:
            if c["parent"]:
                parent = configs[c["parent"]]
                self.assertEqual(
                    sum(
                        c[k] != parent[k]
                        for k in ["head_lr", "lora_lr", "score_brier_weight"]
                    ),
                    1,
                )

    def test_overconfident_errors_improve_brier_without_changing_argmax(self):
        logits = torch.tensor([[8.0, 0.0]] * 10 + [[0.0, 8.0]] * 10)
        targets = torch.tensor([0] * 7 + [1] * 3 + [1] * 7 + [0] * 3)
        temperature, score = fit_temperature(logits, targets)
        self.assertGreater(temperature, 1)
        self.assertLess(score["after"], score["before"])
        ps = probabilities(logits, temperature)
        self.assertTrue(torch.equal(ps.argmax(-1), logits.argmax(-1)))
        torch.testing.assert_close(ps.sum(-1), torch.ones(20))

    def test_mask_variable_classes_and_temperature_validation(self):
        logits = torch.tensor([[2.0, 1.0, -torch.inf], [1.0, 2.0, 0.0]])
        self.assertEqual(probabilities(logits, 2)[0, 2].item(), 0)
        torch.testing.assert_close(probabilities(logits, 1), logits.softmax(-1))
        for invalid in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                probabilities(logits, invalid)
        rows = [
            {"type": kind, "logits": [2.0, 1.0], "target": 0}
            for kind in ["choice", "noul", "score"]
        ]
        rows.append({"type": "score", "logits": [0.0, 1.0, 3.0], "target": 2})
        temperatures, result = fit_by_type(rows)
        self.assertEqual(set(temperatures), {"choice", "noul", "score"})
        self.assertEqual(result["score"]["records"], 2)

    def test_group_split_is_reproducible_and_disjoint(self):
        rows = [
            {
                "id": f"{kind}-{g}-{q}",
                "type": kind,
                "task": kind,
                "group": f"{kind}-{g}",
            }
            for kind in ["choice", "noul", "score"]
            for g in range(4)
            for q in range(2)
        ]
        parts = validation_partition(rows)
        self.assertEqual(
            set(parts["selection"]) | set(parts["calibration"]), {r["id"] for r in rows}
        )
        self.assertFalse(set(parts["selection"]) & set(parts["calibration"]))
        left = {r["group"] for r in rows if r["id"] in parts["selection"]}
        right = {r["group"] for r in rows if r["id"] in parts["calibration"]}
        self.assertFalse(left & right)
        self.assertEqual(
            left,
            {
                r["group"]
                for r in rows
                if r["id"] in validation_partition(rows[::-1])["selection"]
            },
        )

    def test_calibration_rejects_stale_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "head.safetensors").write_bytes(b"original")
            info = {
                "status": "completed",
                "method": "temperature_per_type",
                "temperatures": {"choice": 1, "noul": 2, "score": 3},
                "weight_hash": weight_hash(path),
            }
            (path / "calibration.json").write_text(json.dumps(info))
            self.assertEqual(load_calibration(path, True)["score"], 3)
            (path / "head.safetensors").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "outros pesos"):
                load_calibration(path, True)

    def test_score_weight_only_changes_score_loss_and_gradient(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]], requires_grad=True)
        targets = torch.tensor([0, 1, 1])
        base = decision_loss(logits, targets, 1.0, reduction="none")
        weighted = decision_loss(logits, targets, [1.0, 1.0, 2.0], reduction="none")
        torch.testing.assert_close(base[:2], weighted[:2])
        self.assertGreater(weighted[2].item(), base[2].item())
        weighted.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError):
            decision_loss(logits, targets, [1.0, 2.0])

    def test_validation_selection_ignores_calibration_examples(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()

            def forward(self, batch):
                return torch.tensor([[2.0, 0.0]]) * self.weight

        rows = [
            {
                "id": f"{kind}-{target}",
                "type": kind,
                "input_ids": [1],
                "option_ids": ["0", "1"],
                "option_positions": [[0], [0]],
                "target": target,
            }
            for kind in ["choice", "noul", "score"]
            for target in [0, 1]
        ]
        selected = {r["id"] for r in rows if r["target"] == 0}
        result = validate(Model(), rows, selection_ids=selected)
        self.assertEqual(result["selection"]["mean_type_accuracy"], 1.0)
        self.assertLess(
            result["selection"]["mean_type_brier"], result["mean_type_brier"]
        )


if __name__ == "__main__":
    unittest.main()
