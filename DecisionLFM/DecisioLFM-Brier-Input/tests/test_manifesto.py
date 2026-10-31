import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from manifesto import main


class ManifestTests(unittest.TestCase):
    def test_custom_data_records_and_document_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = {
                "choice": {"type": "choice", "criteria": {"a": "Alpha", "b": "Beta"}},
                "noul": {"type": "noul"},
                "score": {"type": "score", "criteria": ["Low", "High"]},
            }
            for split in ["train", "validation", "test"]:
                rows = []
                for kind, question in questions.items():
                    key = f"{split}_{kind}"
                    rows.append(
                        {
                            "id": key,
                            "task": kind,
                            "type": kind,
                            "group": key,
                            "held_out_task": False,
                            "state": "Example",
                            "questions": {"q": question},
                            "targets": {
                                "q": {"choice": "a", "noul": "true", "score": "1"}[kind]
                            },
                        }
                    )
                (root / f"{split}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n"
                )
            with patch.object(sys, "argv", ["manifesto.py", "--data-dir", str(root)]):
                main()
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(
                manifest["train"]["types"], {"choice": 1, "noul": 1, "score": 1}
            )
            (root / "manifest.json").unlink()
            rows = [
                json.loads(line)
                for line in (root / "validation.jsonl").read_text().splitlines()
            ]
            rows[0]["group"] = "train_choice"
            (root / "validation.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n"
            )
            with (
                patch.object(sys, "argv", ["manifesto.py", "--data-dir", str(root)]),
                self.assertRaises(ValueError),
            ):
                main()


if __name__ == "__main__":
    unittest.main()
