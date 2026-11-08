"""Testes unitários específicos do QLoRA (4-bit NF4, k-bit prep, otimizadores e comparação pareada)."""

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from comparar_qlora import compare_qlora, format_markdown
from modelo import (
    DEFAULT_QUANTIZATION,
    LORA_TARGETS,
    VISION_SKIP_MODULES,
    DecisionLFM,
    make_bnb_config,
    quantization_metadata,
)
from treinar_tipos import build_optimizer


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(hidden_size=hidden_size)
        )
        self.weight = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.gradient_checkpointing_kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs


class DummyTokenizer:
    pad_token_id = 0

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "tokenizer_config.json").write_text("{}\n")


class QLoRATests(unittest.TestCase):
    def test_bnb_config_and_quantization_metadata_validation(self):
        meta = quantization_metadata()
        self.assertEqual(meta, DEFAULT_QUANTIZATION)
        self.assertTrue(meta["load_in_4bit"])
        self.assertEqual(meta["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(meta["bnb_4bit_use_double_quant"])
        self.assertEqual(meta["bnb_4bit_compute_dtype"], "bfloat16")
        self.assertEqual(meta["llm_int8_skip_modules"], list(VISION_SKIP_MODULES))

        unquantized = quantization_metadata(enabled=False)
        self.assertEqual(
            unquantized, {"enabled": False, "method": "none", "dtype": "bfloat16"}
        )

        with self.assertRaises(ValueError):
            quantization_metadata(quant_type="int8")
        with self.assertRaises(ValueError):
            quantization_metadata(compute_dtype="fp8")
        with self.assertRaises(ValueError):
            make_bnb_config(quant_type="bad")
        with self.assertRaises(ValueError):
            make_bnb_config(double_quant="yes")
        with self.assertRaises(ValueError):
            make_bnb_config(compute_dtype="invalid")

        fake_bnb = MagicMock(side_effect=lambda **kwargs: kwargs)
        with patch.dict("sys.modules", {"transformers": types.SimpleNamespace(BitsAndBytesConfig=fake_bnb)}):
            cfg = make_bnb_config(quant_type="nf4", double_quant=True, compute_dtype="bfloat16")
            self.assertTrue(cfg["load_in_4bit"])
            self.assertEqual(cfg["bnb_4bit_quant_type"], "nf4")
            self.assertTrue(cfg["bnb_4bit_use_double_quant"])
            self.assertEqual(cfg["bnb_4bit_compute_dtype"], torch.bfloat16)
            self.assertEqual(cfg["llm_int8_skip_modules"], list(VISION_SKIP_MODULES))

    def test_add_lora_invokes_prepare_model_for_kbit_training_when_quantized(self):
        backbone = DummyBackbone(16)
        model = DecisionLFM(backbone, DummyTokenizer(), head_size=8)
        self.assertTrue(model.is_quantized_4bit)
        self.assertEqual(model.head.query.weight.dtype, torch.float32)

        prepared_backbone = DummyBackbone(16)
        peft_backbone = DummyBackbone(16)
        mock_prepare = MagicMock(return_value=prepared_backbone)
        mock_get_peft = MagicMock(return_value=peft_backbone)
        mock_lora_config = MagicMock(side_effect=lambda **kwargs: kwargs)
        fake_peft = types.SimpleNamespace(
            LoraConfig=mock_lora_config,
            get_peft_model=mock_get_peft,
            prepare_model_for_kbit_training=mock_prepare,
        )
        with patch.dict("sys.modules", {"peft": fake_peft}):
            model.add_lora(rank=8, alpha=16, dropout=0.05)

        mock_prepare.assert_called_once_with(
            backbone,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        mock_lora_config.assert_called_once_with(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=LORA_TARGETS,
        )
        mock_get_peft.assert_called_once()
        self.assertTrue(model.has_lora)

        with self.assertRaises(ValueError):
            model.add_lora(rank=0)
        with self.assertRaises(ValueError):
            model.add_lora(rank=8, dropout=1.0)

    def test_save_and_load_preserve_quantization_and_support_legacy_bf16_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "qlora_ckpt"
            backbone = DummyBackbone(16)
            model = DecisionLFM(backbone, DummyTokenizer(), head_size=8)
            model.save(ckpt)
            saved_cfg = json.loads((ckpt / "decision_config.json").read_text())
            self.assertEqual(saved_cfg["quantization"], DEFAULT_QUANTIZATION)

            created_calls = []

            def fake_create(head_size, max_length, model_id, revision, quantize_4bit=True, **kwargs):
                created_calls.append((quantize_4bit, kwargs))
                return DecisionLFM(
                    DummyBackbone(16),
                    DummyTokenizer(),
                    head_size=head_size,
                    max_length=max_length,
                    quantization=quantization_metadata(enabled=quantize_4bit, **kwargs)
                    if quantize_4bit
                    else quantization_metadata(enabled=False),
                )

            fake_transformers = types.SimpleNamespace(
                AutoTokenizer=types.SimpleNamespace(
                    from_pretrained=lambda *a, **kw: DummyTokenizer()
                )
            )
            fake_peft = types.SimpleNamespace(
                PeftModel=types.SimpleNamespace(from_pretrained=lambda m, p, **kw: m)
            )
            with (
                patch("modelo.create", side_effect=fake_create),
                patch.dict(
                    "sys.modules",
                    {"transformers": fake_transformers, "peft": fake_peft},
                ),
            ):
                loaded_qlora = DecisionLFM.load(ckpt, calibrated=False)
                self.assertTrue(loaded_qlora.is_quantized_4bit)
                self.assertTrue(created_calls[-1][0])
                self.assertEqual(created_calls[-1][1]["quant_type"], "nf4")

                # Legacy checkpoint without "quantization" field defaults to BF16 (use_4bit=False)
                legacy_cfg = {k: v for k, v in saved_cfg.items() if k != "quantization"}
                (ckpt / "decision_config.json").write_text(json.dumps(legacy_cfg))
                loaded_legacy = DecisionLFM.load(ckpt, calibrated=False)
                self.assertFalse(loaded_legacy.is_quantized_4bit)
                self.assertFalse(created_calls[-1][0])

    def test_optimizer_builder_supports_adamw_and_paged_optimizers(self):
        param = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [param], "lr": 1e-4}]
        opt = build_optimizer("adamw", groups)
        self.assertIsInstance(opt, torch.optim.AdamW)

        fake_paged32 = MagicMock(return_value="paged32_instance")
        fake_paged8 = MagicMock(return_value="paged8_instance")
        fake_bnb = types.SimpleNamespace(
            optim=types.SimpleNamespace(
                PagedAdamW32bit=fake_paged32, PagedAdamW8bit=fake_paged8
            )
        )
        with patch.dict("sys.modules", {"bitsandbytes": fake_bnb}):
            self.assertEqual(
                build_optimizer("paged_adamw_32bit", groups), "paged32_instance"
            )
            self.assertEqual(
                build_optimizer("paged_adamw_8bit", groups), "paged8_instance"
            )
        with self.assertRaises(ValueError):
            build_optimizer("sgd", groups)

    def test_compare_qlora_pairs_cohorts_and_reports_training_vram(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lora_ckpt, qlora_ckpt = root / "ckpt_lora", root / "ckpt_qlora"
            lora_ckpt.mkdir()
            qlora_ckpt.mkdir()
            (lora_ckpt / "training.json").write_text(
                json.dumps(
                    {
                        "training_regime": "mixed_choice_noul_score",
                        "selected_epoch": 3,
                        "completed_epochs": 3,
                        "trainable_parameters": 9105409,
                        "elapsed_seconds": 1800.0,
                        "peak_allocated_vram_gib": 4.48,
                    }
                )
            )
            (qlora_ckpt / "training.json").write_text(
                json.dumps(
                    {
                        "training_regime": "qlora_4bit_mixed_choice_noul_score",
                        "quantization": DEFAULT_QUANTIZATION,
                        "optimizer": "paged_adamw_32bit",
                        "selected_epoch": 3,
                        "completed_epochs": 3,
                        "trainable_parameters": 9105409,
                        "elapsed_seconds": 1950.0,
                        "peak_allocated_vram_gib": 2.15,
                    }
                )
            )

            lora_eval, qlora_eval = root / "eval_lora", root / "eval_qlora"
            for directory, ckpt, vram in [
                (lora_eval, lora_ckpt, 3.68),
                (qlora_eval, qlora_ckpt, 1.82),
            ]:
                directory.mkdir()
                run = {
                    "status": "completed",
                    "checkpoint": str(ckpt),
                    "gpu": "RTX 4090",
                    "batch_size": 1,
                    "warmup_per_type": 3,
                    "memory_kind": "dedicated_vram",
                    "expected_records": 3,
                    "data": {"test": {"sha256": "same_test_sha"}},
                }
                rows = [
                    {
                        "id": "c1",
                        "task": "banking77",
                        "type": "choice",
                        "held_out_task": False,
                        "target": "a",
                        "option_ids": ["a", "b"],
                        "prediction": "a",
                        "probabilities": {"a": 0.9, "b": 0.1},
                        "error": None,
                        "latency_ms": 15.0,
                        "input_tokens": 100,
                        "peak_allocated_vram_gib": vram,
                    },
                    {
                        "id": "n1",
                        "task": "boolq",
                        "type": "noul",
                        "held_out_task": False,
                        "target": "true",
                        "option_ids": ["true", "false"],
                        "prediction": "true",
                        "probabilities": {"true": 0.8, "false": 0.2},
                        "error": None,
                        "latency_ms": 12.0,
                        "input_tokens": 80,
                        "peak_allocated_vram_gib": vram,
                    },
                    {
                        "id": "s1",
                        "task": "sst5",
                        "type": "score",
                        "held_out_task": False,
                        "target": "2",
                        "option_ids": ["0", "1", "2"],
                        "prediction": "2",
                        "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
                        "score": 1.6,
                        "error": None,
                        "latency_ms": 14.0,
                        "input_tokens": 90,
                        "peak_allocated_vram_gib": vram,
                    },
                ]
                (directory / "run.json").write_text(json.dumps(run))
                (directory / "predictions.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in rows) + "\n"
                )

            report = compare_qlora(lora_eval, qlora_eval)
            self.assertEqual(report["records"], 3)
            self.assertEqual(report["prediction_agreement_rate"], 1.0)
            self.assertAlmostEqual(
                report["models"]["LoRA-BF16"]["training"]["peak_allocated_vram_gib"],
                4.48,
            )
            self.assertAlmostEqual(
                report["models"]["QLoRA-NF4"]["training"]["peak_allocated_vram_gib"],
                2.15,
            )
            md = format_markdown(report)
            self.assertIn("QLoRA (4-bit NF4)", md)
            self.assertIn("Pico de VRAM no treino", md)


if __name__ == "__main__":
    unittest.main()

