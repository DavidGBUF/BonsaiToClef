"""Avalia JSONL tipado, batch 1, com latência, qualidade, VRAM e falhas."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from dados_tipos import digest, read_jsonl
from esquema import format_answer, prepare_record
from modelo import DecisionLFM, checkpoint_hash
from metricas import metrics


CLEF_MODEL = "Cloudflare/clef-flash"
CLEF_REVISION = "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"


def load_clef(revision, directory=None):
    from huggingface_hub import snapshot_download

    path = (
        str(directory.resolve())
        if directory
        else snapshot_download(
            CLEF_MODEL, revision=revision, token=False, local_files_only=True
        )
    )
    sys.path.insert(0, path)
    from joint_schema_model import collate_records, encode_record, load_release_model

    model, processor = load_release_model(path, device="cuda", dtype=torch.bfloat16)

    @torch.inference_mode()
    def responder(sample):
        _, questions = prepare_record(sample)
        record = {"state": sample["state"], "questions": sample["questions"]}
        # Preserve all input: the official encoder otherwise truncates the state.
        encoded = encode_record(
            processor.tokenizer, record, processor=processor, max_length=sys.maxsize
        )
        if len(encoded.input_ids) > 8192:
            raise ValueError(
                f"Entrada com {len(encoded.input_ids)} tokens; limite 8192."
            )
        batch = collate_records(
            [encoded], processor.tokenizer.pad_token_id, torch.device("cuda")
        )
        logits = model(batch)[0]
        answers = {}
        for question, values in zip(encoded.questions, logits):
            probabilities = dict(
                zip(question.option_ids, values.float().softmax(-1).tolist())
            )
            # Use full precision, matching DecisionLFM, instead of SystemOne's four-digit rounding.
            answers[question.question_id] = format_answer(
                questions[question.question_id], probabilities
            )
        return {
            "answers": answers,
            "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
        }

    return model, responder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    model_args = parser.add_mutually_exclusive_group(required=True)
    model_args.add_argument("--checkpoint", type=Path)
    model_args.add_argument("--clef", action="store_true")
    parser.add_argument("--clef-revision", default=CLEF_REVISION)
    parser.add_argument("--clef-dir", type=Path, help="Snapshot local completo do CLEF")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--uncalibrated", action="store_true", help="Controle com T=1, mesmos pesos"
    )
    args = parser.parse_args()
    if args.output_dir.exists() or args.warmup < 0:
        parser.error("Use diretório novo e warmup >= 0")
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if digest(args.data_dir / "test.jsonl") != manifest["test"]["sha256"]:
        raise ValueError("Teste diferente do manifesto")
    samples = read_jsonl(args.data_dir / "test.jsonl")
    test_groups = {row["group"] for row in samples}
    for split in ["train", "validation"]:
        if test_groups & {
            row["group"] for row in read_jsonl(args.data_dir / f"{split}.jsonl")
        }:
            raise ValueError("Documento de teste presente no treino/validação")
    if not samples or any(
        len(row["questions"]) != 1 or len(row["targets"]) != 1 for row in samples
    ):
        raise ValueError("O benchmark requer uma pergunta e um alvo por registro")
    torch.set_num_threads(8)
    if args.clef:
        model, responder = load_clef(args.clef_revision, args.clef_dir)
        model_info = {
            "model": CLEF_MODEL,
            "model_revision": args.clef_revision,
            "training_regime": "zero_shot_on_this_test",
            "dtype": "backbone BF16; head BF16",
            "source_dir": str(args.clef_dir.resolve()) if args.clef_dir else None,
        }
    else:
        training = json.loads((args.checkpoint / "training.json").read_text())
        if training["status"] != "completed":
            raise ValueError("Checkpoint de treinamento incompleto")
        if training.get("training_regime") == "mixed_choice_noul_score" and (
            training["data_manifest_sha256"] != digest(args.data_dir / "manifest.json")
        ):
            raise ValueError("Manifesto de dados diferente do treino")
        model = DecisionLFM.load(args.checkpoint, calibrated=not args.uncalibrated)
        responder = model.responder
        model_info = {
            "model": "DecisionLFM",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_hash": checkpoint_hash(args.checkpoint),
            "training_regime": training.get("training_regime", "supervised_banking77"),
            "dtype": "backbone BF16; head FP32",
            "temperatures": model.temperatures,
            "selected_epoch": training.get("selected_epoch"),
        }
    device = next(model.parameters()).device
    backend = torch.cuda if device.type == "cuda" else torch.mps
    for kind in ["choice", "noul", "score"]:
        first = next((row for row in samples if row["type"] == kind), None)
        if first is None:
            continue
        for _ in range(args.warmup):
            responder(first)
    args.output_dir.mkdir(parents=True)
    run = {
        "status": "running",
        **model_info,
        "data": manifest,
        "batch_size": 1,
        "warmup_per_type": args.warmup,
        "gpu": torch.cuda.get_device_name()
        if device.type == "cuda"
        else "Apple Silicon (MPS)",
        "device": str(device),
        "memory_kind": "dedicated_vram" if device.type == "cuda" else "unified_mps",
        "expected_records": len(samples),
        "max_length": 8192,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "numpy", "scikit-learn", "tokenizers"]
        },
        "protocol": "Latency includes JSON normalization, encoding, transfer, backbone, head and probability "
        "formatting with CUDA synchronization; excludes loading, warmup and disk writes. "
        "Brier=sum over all allowed options, range [0,2], no factor 1/2. "
        "Failures count as wrong for accuracy/F1; probability, latency and VRAM use valid calls. "
        "Macro-F1 for aggregates is the unweighted mean of per-task macro-F1. "
        "Score accuracy/F1 use argmax level; MAE uses expected score. VRAM is peak torch allocation, including weights.",
        "code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
    }
    if device.type == "mps":
        run["protocol"] = (
            run["protocol"]
            .replace("CUDA synchronization", "MPS synchronization")
            .replace(
                "VRAM is peak torch allocation, including weights.",
                "Dedicated VRAM is not applicable. MPS memory is the maximum observed current allocation "
                "after synchronized calls, including weights; this is not the peak inside each call.",
            )
        )
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    rows = []
    with (args.output_dir / "predictions.jsonl").open("w") as output:
        for index, sample in enumerate(samples):
            _, questions = prepare_record(sample)
            key, question = next(iter(questions.items()))
            row = {
                "id": sample["id"],
                "task": sample["task"],
                "type": sample["type"],
                "held_out_task": sample["held_out_task"],
                "target": str(sample["targets"][key]),
                "option_ids": list(question["options"]),
                "prediction": None,
                "probabilities": None,
                "error": None,
                "latency_ms": None,
                "peak_allocated_vram_gib": None,
            }
            try:
                backend.synchronize()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                response = responder(sample)
                answer = response["answers"][key]
                probabilities = (
                    {"true": answer["noul"], "false": 1 - answer["noul"]}
                    if sample["type"] == "noul"
                    else answer["probabilities"]
                )
                if (
                    set(probabilities) != set(row["option_ids"])
                    or not all(
                        np.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()
                    )
                    or not np.isclose(sum(probabilities.values()), 1, atol=1e-5)
                ):
                    raise ValueError("Distribuição inválida")
                backend.synchronize()
                row.update(
                    prediction=max(probabilities, key=probabilities.get),
                    probabilities=probabilities,
                    input_tokens=response["usage"]["input_tokens"],
                    latency_ms=(time.perf_counter() - started) * 1000,
                    peak_allocated_vram_gib=(
                        torch.cuda.max_memory_allocated() / 1024**3
                        if device.type == "cuda"
                        else None
                    ),
                )
                if device.type == "mps":
                    row["mps_allocated_gib"] = (
                        torch.mps.current_allocated_memory() / 1024**3
                    )
                if sample["type"] == "score":
                    row["score"] = answer["score"]
            except (RuntimeError, ValueError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
                backend.empty_cache()
            rows.append(row)
            output.write(json.dumps(row, allow_nan=False) + "\n")
            output.flush()
            if (index + 1) % 100 == 0 or index + 1 == len(samples):
                print(
                    f"test={index + 1}/{len(samples)} failures={sum(r['error'] is not None for r in rows)}",
                    flush=True,
                )
    result = {
        "all": metrics(rows),
        "by_task": {},
        "in_domain": {},
        "held_out_tasks": {},
    }
    for task in sorted({row["task"] for row in rows}):
        result["by_task"][task] = metrics([row for row in rows if row["task"] == task])
    for scope, held_out in [("in_domain", False), ("held_out_tasks", True)]:
        for kind in ["choice", "noul", "score"]:
            result[scope][kind] = metrics(
                [
                    r
                    for r in rows
                    if r["held_out_task"] == held_out and r["type"] == kind
                ]
            )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    run.update(
        status="completed",
        processed_records=len(rows),
        failed_records=result["all"]["failed_records"],
    )
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
