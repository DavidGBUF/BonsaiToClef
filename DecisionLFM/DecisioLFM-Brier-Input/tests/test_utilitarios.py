import tempfile
from pathlib import Path
import unittest

from metricas import metrics
from modelo import checkpoint_hash
from dados_tipos import code_hashes, digest


class UtilityTests(unittest.TestCase):
    def test_experiment_hashes_only_include_project_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = ["modelo.py", "gguf/runtime.py", "tests/test_modelo.py"]
            external = [
                ".venv/lib/site-packages/torch/__init__.py",
                "backends/llama.cpp/convert.py",
                "resultados/run/codigo/modelo.py",
                "modelos/checkpoint/config.py",
            ]
            for name in sources + external:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            hashes = code_hashes(root)
            self.assertEqual(set(hashes), set(sources))
            self.assertEqual(hashes["modelo.py"], digest(root / "modelo.py"))

    def test_final_checkpoint_hash_excludes_intermediate_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "head.safetensors").write_bytes(b"final")
            initial = checkpoint_hash(root)
            intermediate = root / "checkpoints/epoch-1"
            intermediate.mkdir(parents=True)
            (intermediate / "head.safetensors").write_bytes(b"intermediate")
            self.assertEqual(checkpoint_hash(root), initial)
            (root / "head.safetensors").write_bytes(b"changed")
            self.assertNotEqual(checkpoint_hash(root), initial)

    def test_empty_metric_scope_is_explicit(self):
        result = metrics([])
        self.assertEqual(result["records"], 0)
        self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["brier_multiclass"])


if __name__ == "__main__":
    unittest.main()
