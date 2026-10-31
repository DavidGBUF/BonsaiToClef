"""Treina uma cabeça compartilhada e um LoRA nos três tipos por N épocas totais."""

import argparse
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

import torch

from dados_tipos import digest, encoded_record, read_jsonl
from modelo import MODEL, REVISION, collate, create
from objetivo import decision_loss


@torch.no_grad()
def validate(model, records, brier_weight=1.0):
    model.eval()
    device = next(model.parameters()).device
    totals = {
        kind: {"correct": 0, "records": 0, "loss": 0.0}
        for kind in ["choice", "noul", "score"]
    }
    for index, record in enumerate(records):
        batch = collate([record], model.tokenizer.pad_token_id, device)
        logits = model(batch)
        target = torch.tensor([record["target"]], device=device)
        loss = decision_loss(logits, target, brier_weight).item()
        if not torch.isfinite(logits[batch["option_mask"]]).all() or not torch.isfinite(
            torch.tensor(loss)
        ):
            raise ValueError("Validação não finita")
        item = totals[record["type"]]
        item["correct"] += int(logits.argmax(-1).item() == record["target"])
        item["records"] += 1
        item["loss"] += loss
        if device.type == "mps" and (index + 1) % 300 == 0:
            print(f"validation={index + 1}/{len(records)}", flush=True)
    for item in totals.values():
        item["accuracy"] = item["correct"] / item["records"]
        item["loss"] /= item["records"]
    return {
        "types": totals,
        "mean_type_accuracy": sum(item["accuracy"] for item in totals.values()) / 3,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--brier-weight",
        type=float,
        default=1.0,
        help="Peso do Brier somado à entropia cruzada; 0 reproduz CE",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Salva pesos e estado a cada N atualizações; 0 desativa",
    )
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or min(
            args.epochs,
            args.batch_size,
            args.gradient_accumulation,
            args.lora_rank,
            args.head_lr,
            args.lora_lr,
        )
        <= 0
    ):
        parser.error("Use diretório novo e parâmetros positivos")
    if not math.isfinite(args.brier_weight) or args.brier_weight < 0:
        parser.error("brier-weight deve ser finito e >= 0")
    if args.checkpoint_every < 0:
        parser.error("checkpoint-every deve ser >= 0")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    samples = {
        split: read_jsonl(args.data_dir / f"{split}.jsonl")
        for split in ["train", "validation"]
    }
    for split in samples:
        if digest(args.data_dir / f"{split}.jsonl") != manifest[split]["sha256"]:
            raise ValueError(f"Dados alterados: {split}")
        if any(row["held_out_task"] for row in samples[split]):
            raise ValueError("Tarefa reservada encontrada no treino/validação")
        if {row["type"] for row in samples[split]} != {"choice", "noul", "score"}:
            raise ValueError(
                f"{split} deve conter exemplos dos três tipos: choice, noul e score"
            )
    if {row["group"] for row in samples["train"]} & {
        row["group"] for row in samples["validation"]
    }:
        raise ValueError("Documento presente no treino e validação")
    model = create()
    device = next(model.parameters()).device
    model.add_lora(args.lora_rank)
    rng = random.Random(args.seed)
    records = {
        split: [
            encoded_record(
                model.tokenizer,
                row,
                manifest["max_input_tokens"],
                rng if split == "train" else None,
            )
            for row in rows
        ]
        for split, rows in samples.items()
    }
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": args.head_lr},
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": args.lora_lr,
            },
        ]
    )
    args.output_dir.mkdir(parents=True)
    training = {
        "status": "running",
        "base_model": MODEL,
        "base_revision": REVISION,
        "training_regime": "mixed_choice_noul_score",
        "epochs": args.epochs,
        "objective": {
            "name": "cross_entropy_plus_brier",
            "brier_weight": args.brier_weight,
            "label_smoothing": 0.0,
            "brier_scale": "sum over classes, range [0,2]",
        },
        "checkpoint_rule": "Final weights after exactly N complete joint head+LoRA epochs",
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "data_manifest_sha256": digest(args.data_dir / "manifest.json"),
        "data": manifest,
        "records": {split: len(rows) for split, rows in samples.items()},
        "types": dict(Counter(row["type"] for row in samples["train"])),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "gpu": torch.cuda.get_device_name()
        if device.type == "cuda"
        else "Apple Silicon (MPS)",
        "device": str(device),
        "history": [],
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "transformers",
                "peft",
                "datasets",
                "numpy",
                "scikit-learn",
            ]
        },
        "code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
    }

    def save_manifest():
        (args.output_dir / "training.json").write_text(
            json.dumps(training, indent=2, allow_nan=False) + "\n"
        )

    save_manifest()
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Nearby lengths share a batch; batch order is reshuffled every epoch.
    ordered = sorted(records["train"], key=lambda r: len(r["input_ids"]))
    batches = [
        ordered[i : i + args.batch_size]
        for i in range(0, len(ordered), args.batch_size)
    ]
    record_index = {id(record): index for index, record in enumerate(records["train"])}

    def save_intermediate(epoch, batch_index, steps, loss_sum, count):
        path = args.output_dir / "checkpoints" / f"epoch-{epoch}-batch-{batch_index}"
        model.save(path)
        state = {
            "epoch": epoch,
            "completed_batches": batch_index,
            "optimizer_steps": steps,
            "loss_sum": loss_sum,
            "records": count,
            "optimizer": optimizer.state_dict(),
            "python_rng": random.getstate(),
            "shuffle_rng": rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "accelerator_rng": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else torch.mps.get_rng_state()
            ),
            "batch_order": [
                [record_index[id(row)] for row in batch] for batch in batches
            ],
            "data_manifest_sha256": training["data_manifest_sha256"],
        }
        temporary = path / "training_state.pt.tmp"
        torch.save(state, temporary)
        temporary.replace(path / "training_state.pt")
        training["last_checkpoint"] = str(path)
        save_manifest()
        print(f"checkpoint={path}", flush=True)

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            rng.shuffle(batches)
            loss_sum = count = steps = accumulated = 0
            optimizer.zero_grad(set_to_none=True)
            for index, items in enumerate(batches):
                batch = collate(items, model.tokenizer.pad_token_id, device)
                target = torch.tensor([row["target"] for row in items], device=device)
                loss = decision_loss(
                    model(batch), target, args.brier_weight, reduction="sum"
                )
                if not torch.isfinite(loss):
                    raise ValueError("Loss não finita")
                loss.backward()
                loss_sum += loss.item()
                count += len(items)
                accumulated += len(items)
                if (index + 1) % args.gradient_accumulation == 0 or index + 1 == len(
                    batches
                ):
                    for parameter in trainable:
                        if parameter.grad is not None:
                            parameter.grad.div_(accumulated)
                    torch.nn.utils.clip_grad_norm_(
                        trainable, 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                    steps += 1
                    if args.checkpoint_every and steps % args.checkpoint_every == 0:
                        save_intermediate(epoch, index + 1, steps, loss_sum, count)
                if (index + 1) % (
                    10 if device.type == "mps" else 100
                ) == 0 or index + 1 == len(batches):
                    print(
                        f"epoch={epoch}/{args.epochs} batch={index + 1}/{len(batches)} "
                        f"loss={loss_sum / count:.5f} elapsed_s={time.perf_counter() - started:.1f}",
                        flush=True,
                    )
            result = validate(model, records["validation"], args.brier_weight)
            training["history"].append(
                {
                    "epoch": epoch,
                    "train_loss": loss_sum / count,
                    "optimizer_steps": steps,
                    "validation": result,
                }
            )
            model.save(args.output_dir)
            save_manifest()
            print(json.dumps(training["history"][-1]), flush=True)
        training["status"] = "completed"
    finally:
        if training["status"] != "completed":
            training["status"] = "interrupted"
        training["elapsed_seconds"] = time.perf_counter() - started
        training["peak_allocated_vram_gib"] = (
            torch.cuda.max_memory_allocated() / 1024**3
            if device.type == "cuda"
            else None
        )
        if device.type == "mps":
            training["final_mps_allocated_gib"] = (
                torch.mps.current_allocated_memory() / 1024**3
            )
        save_manifest()


if __name__ == "__main__":
    main()
