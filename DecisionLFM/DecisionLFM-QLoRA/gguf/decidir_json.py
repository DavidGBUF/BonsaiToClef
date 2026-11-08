"""Lê o mesmo JSON da versão PyTorch e responde com DecisionLFM GGUF."""

import argparse
import json
from pathlib import Path

from runtime import Runtime, prepare_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--capacity", type=int, default=2048)
    args = parser.parse_args()
    if args.capacity < 1:
        parser.error("capacity deve ser > 0")
    try:
        sample = json.loads(args.input.read_text(encoding="utf-8"))
        prepare_record(sample)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    runtime = Runtime(
        "decision",
        args.backbone,
        args.head,
        args.tokenizer,
        args.capacity,
        args.library,
    )
    try:
        print(
            json.dumps(
                runtime.respond(sample), ensure_ascii=False, indent=2, allow_nan=False
            )
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

