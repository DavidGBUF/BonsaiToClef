"""Recarrega os pesos finais e repete a validação registrada no treinamento."""

import argparse
import json
from pathlib import Path

import torch

from dados_tipos import digest, encoded_record, read_jsonl
from modelo import DecisionLFM
from treinar_tipos import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use um arquivo de saída novo")
    training = json.loads((args.checkpoint / "training.json").read_text())
    if training["status"] != "completed" or not training["history"]:
        raise ValueError("Treinamento incompleto")
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if training["data_manifest_sha256"] != digest(
        args.data_dir / "manifest.json"
    ) or manifest["validation"]["sha256"] != digest(args.data_dir / "validation.jsonl"):
        raise ValueError("Validação diferente da usada no treinamento")
    torch.set_num_threads(8)
    model = DecisionLFM.load(args.checkpoint, calibrated=False)
    records = [
        encoded_record(model.tokenizer, row, manifest["max_input_tokens"])
        for row in read_jsonl(args.data_dir / "validation.jsonl")
    ]
    partition = training.get("validation_partition")
    actual = validate(
        model,
        records,
        training["objective"]["brier_weight"],
        training["objective"].get("score_brier_weight"),
        set(partition["selection"]) if partition else None,
    )
    selected = training.get("selected_epoch", training["history"][-1]["epoch"])
    expected = next(
        h["validation"] for h in training["history"] if h["epoch"] == selected
    )
    for kind in ["choice", "noul", "score"]:
        a, b = actual["types"][kind], expected["types"][kind]
        if (
            a["records"] != b["records"]
            or a["correct"] != b["correct"]
            or abs(a["loss"] - b["loss"]) > 1e-5
        ):
            raise ValueError(f"Validação recarregada divergiu para {kind}: {a} vs {b}")
    result = {
        "status": "passed",
        "records": len(records),
        "selected_epoch": selected,
        "quantization": model.quantization,
        "actual": actual,
        "expected": expected,
        "max_abs_loss_difference": max(
            abs(actual["types"][k]["loss"] - expected["types"][k]["loss"])
            for k in actual["types"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

