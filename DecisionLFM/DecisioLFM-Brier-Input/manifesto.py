"""Valida JSONL próprio e cria o manifesto consumido pelo treino e avaliação."""

import argparse
from collections import Counter
import json
from pathlib import Path

from dados_tipos import digest, read_jsonl
from esquema import prepare_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    output = args.data_dir / "manifest.json"
    if output.exists() or args.max_length < 1:
        parser.error("Use manifesto novo e max-length > 0")
    manifest = {"max_input_tokens": args.max_length, "sources": {}}
    groups, ids = set(), set()
    for split in ["train", "validation", "test"]:
        path = args.data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"Split vazio: {split}")
        local_groups = set()
        for row in rows:
            required = {
                "id",
                "task",
                "type",
                "group",
                "held_out_task",
                "state",
                "questions",
                "targets",
            }
            if required - row.keys():
                raise ValueError(
                    f"Campos ausentes em {split}: {sorted(required - row.keys())}"
                )
            _, questions = prepare_record(row)
            if len(questions) != 1 or set(row["targets"]) != set(questions):
                raise ValueError(
                    "Treino/avaliação requerem uma pergunta e seu alvo por registro"
                )
            key, question = next(iter(questions.items()))
            if (
                row["type"] != question["type"]
                or str(row["targets"][key]) not in question["options"]
            ):
                raise ValueError(f"Tipo ou alvo inválido: {row['id']}")
            if row["id"] in ids or row["group"] in groups:
                raise ValueError("ID repetido ou documento compartilhado entre splits")
            if split != "test" and row["held_out_task"]:
                raise ValueError("Tarefa reservada no treino/validação")
            ids.add(row["id"])
            local_groups.add(row["group"])
        groups.update(local_groups)
        manifest[split] = {
            "records": len(rows),
            "sha256": digest(path),
            "types": dict(Counter(row["type"] for row in rows)),
        }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
