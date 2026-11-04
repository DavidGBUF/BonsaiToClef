"""Ajusta temperaturas com logits de validação; não modifica cabeça ou LoRA."""

import argparse
import json
from pathlib import Path

import torch

from calibracao import fit_by_type, summarize_logits, validation_partition, weight_hash
from dados_tipos import digest, encoded_record, read_jsonl
from modelo import DecisionLFM, collate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--protect-score-mae", action="store_true")
    args = p.parse_args()
    output = args.checkpoint / "calibration.json"
    if output.exists():
        p.error("Calibração já existe; preserve o resultado anterior")
    training = json.loads((args.checkpoint / "training.json").read_text())
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if training["status"] != "completed":
        raise ValueError("Treinamento incompleto")
    if training["data_manifest_sha256"] != digest(args.data_dir / "manifest.json"):
        raise ValueError("Manifesto diferente do treino")
    if digest(args.data_dir / "validation.jsonl") != manifest["validation"]["sha256"]:
        raise ValueError("Validação alterada")
    samples = read_jsonl(args.data_dir / "validation.jsonl")
    partition = training.get("validation_partition") or validation_partition(
        samples, training["args"]["seed"]
    )
    keep = set(partition["calibration"])
    selection_ids = set(partition["selection"])
    if keep & selection_ids:
        raise ValueError("Partições sobrepostas")
    torch.set_num_threads(8)
    model = DecisionLFM.load(args.checkpoint, calibrated=False)
    device = next(model.parameters()).device
    rows = []
    with torch.inference_mode():
        for sample in samples:
            record = encoded_record(
                model.tokenizer, sample, manifest["max_input_tokens"]
            )
            batch = collate([record], model.tokenizer.pad_token_id, device)
            logits = model(batch)[0].float().cpu().tolist()
            rows.append(
                {
                    "id": sample["id"],
                    "type": sample["type"],
                    "logits": logits,
                    "target": record["target"],
                }
            )
    calibration_rows = [r for r in rows if r["id"] in keep]
    selection_rows = [r for r in rows if r["id"] in selection_ids]
    if {r["id"] for r in calibration_rows} != keep:
        raise ValueError("IDs de calibração ausentes da validação")
    temperatures, metrics = fit_by_type(calibration_rows, args.protect_score_mae)
    info = {
        "status": "completed",
        "method": "temperature_per_type",
        "objective": "brier_sum_[0,2]",
        "temperatures": temperatures,
        "bounds": [0.25, 16.0],
        "score_mae_protected": args.protect_score_mae,
        "by_type": metrics,
        "weight_hash": weight_hash(args.checkpoint),
        "validation_sha256": manifest["validation"]["sha256"],
        "selection_ids": partition["selection"],
        "calibration_ids": sorted(keep),
        "selection_before": summarize_logits(selection_rows),
        "selection_after": summarize_logits(selection_rows, temperatures),
        "protocol": "Fit only on calibration half of validation; selection half chooses epochs/hyperparameters. Test never read.",
    }
    (args.checkpoint / "calibration-logits.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in calibration_rows)
    )
    (args.checkpoint / "validation-logits.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
    )
    output.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({"temperatures": temperatures, "by_type": metrics}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
