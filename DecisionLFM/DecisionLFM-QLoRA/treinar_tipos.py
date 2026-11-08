"""Treina até três épocas com QLoRA (4-bit NF4 + Double Quant) e seleciona por validação."""

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
from modelo import MODEL, REVISION, VISION_SKIP_MODULES, collate, create
from objetivo import decision_loss
from calibracao import TYPES, validation_partition

OPTIMIZERS = ("adamw", "paged_adamw_32bit", "paged_adamw_8bit")


def build_optimizer(name, param_groups):
    if name == "adamw":
        return torch.optim.AdamW(param_groups)
    if name in {"paged_adamw_32bit", "paged_adamw_8bit"}:
        import bitsandbytes as bnb

        cls = (
            bnb.optim.PagedAdamW32bit
            if name == "paged_adamw_32bit"
            else bnb.optim.PagedAdamW8bit
        )
        return cls(param_groups)
    raise ValueError(f"Otimizador desconhecido: {name}")


@torch.no_grad()
def validate(
    model, records, brier_weight=1.0, score_brier_weight=None, selection_ids=None
):
    model.eval()
    device = next(model.head.parameters()).device

    def empty():
        return {
            kind: {"correct": 0, "records": 0, "loss": 0.0, "brier": 0.0, "mae": 0.0}
            for kind in TYPES
        }

    totals, selected = empty(), empty()
    score_brier_weight = (
        brier_weight if score_brier_weight is None else score_brier_weight
    )
    for index, record in enumerate(records):
        batch = collate([record], model.tokenizer.pad_token_id, device)
        logits = model(batch)
        target = torch.tensor([record["target"]], device=device)
        weight = score_brier_weight if record["type"] == "score" else brier_weight
        loss = decision_loss(logits, target, weight).item()
        if not torch.isfinite(logits[batch["option_mask"]]).all() or not math.isfinite(
            loss
        ):
            raise ValueError("Validação não finita")
        ps = logits.float().softmax(-1)[0]
        brier = (ps.square().sum() - 2 * ps[record["target"]] + 1).item()
        expected_score = (ps * torch.arange(len(ps), device=device)).sum().item()
        collections = [totals]
        if selection_ids is not None and record["id"] in selection_ids:
            collections.append(selected)
        for collection in collections:
            item = collection[record["type"]]
            item["correct"] += int(logits.argmax(-1).item() == record["target"])
            item["records"] += 1
            item["loss"] += loss
            item["brier"] += brier
            item["mae"] += abs(expected_score - record["target"])
        if device.type == "mps" and (index + 1) % 300 == 0:
            print(f"validation={index + 1}/{len(records)}", flush=True)

    def summarize(values):
        for kind, item in values.items():
            item["accuracy"] = item["correct"] / item["records"]
            for key in ["loss", "brier", "mae"]:
                item[key] /= item["records"]
            if kind != "score":
                item.pop("mae")
        return {
            "types": values,
            "mean_type_accuracy": sum(item["accuracy"] for item in values.values()) / 3,
            "mean_type_brier": sum(item["brier"] for item in values.values()) / 3,
        }

    result = summarize(totals)
    if selection_ids is not None:
        result["selection"] = summarize(selected)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--lora-lr", type=float, default=3e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="Escala alpha do LoRA; padrão = 2 * lora-rank",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--quant-type",
        choices=["nf4", "fp4"],
        default="nf4",
        help="Tipo de dados de 4 bits do QLoRA (padrão: nf4)",
    )
    parser.add_argument(
        "--no-double-quant",
        action="store_true",
        help="Desativa a Dupla Quantização (bnb_4bit_use_double_quant=False)",
    )
    parser.add_argument(
        "--no-qlora",
        action="store_true",
        help="Carrega o backbone em BF16 sem quantização 4-bit (controle LoRA padrão)",
    )
    parser.add_argument(
        "--optimizer",
        choices=list(OPTIMIZERS),
        default="adamw",
        help="adamw (PyTorch) ou paged_adamw_32bit / paged_adamw_8bit (bitsandbytes)",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--brier-weight",
        type=float,
        default=1.0,
        help="Peso do Brier somado à entropia cruzada; 0 reproduz CE",
    )
    parser.add_argument(
        "--score-brier-weight",
        type=float,
        default=2.0,
        help="Peso do Brier para score; os outros tipos usam --brier-weight",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument(
        "--selection-metric", choices=["brier", "accuracy"], default="brier"
    )
    parser.add_argument(
        "--lr-scheduler", choices=["plateau", "constant"], default="plateau"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Salva pesos e estado a cada N atualizações; 0 desativa",
    )
    args = parser.parse_args()
    lora_alpha = 2 * args.lora_rank if args.lora_alpha is None else args.lora_alpha
    if (
        args.output_dir.exists()
        or min(
            args.epochs,
            args.batch_size,
            args.gradient_accumulation,
            args.lora_rank,
            lora_alpha,
            args.head_lr,
            args.lora_lr,
            args.max_grad_norm,
        )
        <= 0
    ):
        parser.error("Use diretório novo e parâmetros positivos")
    if not (0.0 <= args.lora_dropout < 1.0):
        parser.error("lora-dropout deve estar em [0, 1)")
    if not math.isfinite(args.brier_weight) or args.brier_weight < 0:
        parser.error("brier-weight deve ser finito e >= 0")
    if not math.isfinite(args.score_brier_weight) or args.score_brier_weight < 0:
        parser.error("score-brier-weight deve ser finito e >= 0")
    if (
        args.epochs > 3
        or args.early_stopping_patience < 1
        or not 0 < args.lr_factor < 1
    ):
        parser.error("Use no máximo três épocas, patience >= 1 e lr-factor entre 0 e 1")
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
    partition = validation_partition(samples["validation"], args.seed)
    selection_ids = set(partition["selection"])
    quantize_4bit = not args.no_qlora
    model = create(
        quantize_4bit=quantize_4bit,
        quant_type=args.quant_type,
        double_quant=not args.no_double_quant,
        compute_dtype="bfloat16",
        skip_modules=VISION_SKIP_MODULES,
    )
    device = next(model.head.parameters()).device
    model.add_lora(
        rank=args.lora_rank, alpha=lora_alpha, dropout=args.lora_dropout
    )
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
    optimizer = build_optimizer(
        args.optimizer,
        [
            {"params": model.head.parameters(), "lr": args.head_lr},
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": args.lora_lr,
            },
        ],
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if args.selection_metric == "brier" else "max",
            factor=args.lr_factor,
            patience=0,
            threshold=1e-4,
            threshold_mode="abs",
            min_lr=[args.head_lr / 16, args.lora_lr / 16],
        )
        if args.lr_scheduler == "plateau"
        else None
    )
    package_names = [
        "torch",
        "transformers",
        "peft",
        "datasets",
        "numpy",
        "scikit-learn",
    ]
    if quantize_4bit or args.optimizer != "adamw":
        package_names.append("bitsandbytes")
    args.output_dir.mkdir(parents=True)
    training = {
        "status": "running",
        "base_model": MODEL,
        "base_revision": REVISION,
        "training_regime": (
            "qlora_4bit_mixed_choice_noul_score"
            if quantize_4bit
            else "mixed_choice_noul_score"
        ),
        "quantization": model.quantization,
        "lora": {
            "rank": args.lora_rank,
            "alpha": lora_alpha,
            "dropout": args.lora_dropout,
        },
        "optimizer": args.optimizer,
        "max_grad_norm": args.max_grad_norm,
        "epochs": args.epochs,
        "objective": {
            "name": "cross_entropy_plus_brier",
            "brier_weight": args.brier_weight,
            "score_brier_weight": args.score_brier_weight,
            "label_smoothing": 0.0,
            "brier_scale": "sum over classes, range [0,2]",
        },
        "checkpoint_rule": (
            "Lowest Brier"
            if args.selection_metric == "brier"
            else "Highest accuracy; Brier breaks ties"
        )
        + " on selection half of validation; at most 3 epochs",
        "validation_partition": partition,
        "selection_metric": "selection.mean_type_" + args.selection_metric,
        "selected_epoch": None,
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
            name: importlib.metadata.version(name) for name in package_names
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
            "scheduler": scheduler.state_dict() if scheduler else None,
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

    best_key, best_brier, bad_epochs = None, math.inf, 0
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
                    model(batch),
                    target,
                    [
                        args.score_brier_weight
                        if row["type"] == "score"
                        else args.brier_weight
                        for row in items
                    ],
                    reduction="sum",
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
                        trainable, args.max_grad_norm, error_if_nonfinite=True
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
            result = validate(
                model,
                records["validation"],
                args.brier_weight,
                args.score_brier_weight,
                selection_ids,
            )
            selection = result["selection"]
            score = selection["mean_type_brier"]
            accuracy = selection["mean_type_accuracy"]
            key = (-score,) if args.selection_metric == "brier" else (accuracy, -score)
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            if scheduler:
                scheduler.step(score if args.selection_metric == "brier" else accuracy)
            training["history"].append(
                {
                    "epoch": epoch,
                    "train_loss": loss_sum / count,
                    "optimizer_steps": steps,
                    "validation": result,
                    "learning_rates": learning_rates,
                    "next_learning_rates": [
                        group["lr"] for group in optimizer.param_groups
                    ],
                }
            )
            best_brier = min(best_brier, score)
            training["best_validation_brier"] = best_brier
            if best_key is None or key > best_key:
                best_key, bad_epochs = key, 0
                training.update(
                    selected_epoch=epoch,
                    selected_validation_accuracy=accuracy,
                    selected_validation_brier=score,
                )
                model.save(args.output_dir)
            else:
                bad_epochs += 1
            training["completed_epochs"] = epoch
            save_manifest()
            print(json.dumps(training["history"][-1]), flush=True)
            if bad_epochs >= args.early_stopping_patience:
                training["stop_reason"] = (
                    f"early_stopping_validation_{args.selection_metric}"
                )
                break
        training.setdefault("stop_reason", "epoch_budget")
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

