"""Lê state e questions de um JSON e imprime as respostas tipadas."""

import argparse
import json
from pathlib import Path

from esquema import prepare_record
from modelo import DecisionLFM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = json.loads(args.input.read_text(encoding="utf-8"))
        prepare_record(record)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    model = DecisionLFM.load(args.checkpoint)
    print(
        json.dumps(
            model.responder(record), ensure_ascii=False, indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()

