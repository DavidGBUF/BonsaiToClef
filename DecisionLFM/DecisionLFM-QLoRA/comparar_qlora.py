"""Compara pareado DecisionLFM (LoRA BF16) vs DecisionLFM-QLoRA (4-bit NF4).

Audita os mesmos exemplos de teste, protocolo, contadores de memória,
métricas por tipo/tarefa (incluindo MAE de score) e estatísticas de treinamento
(pico de VRAM e tempo de treino quando training.json dos checkpoints é informado).
"""

import argparse
import json
from pathlib import Path

from calibracao import TYPES
from dados_tipos import read_jsonl
from metricas import metrics


def fmt(value, digits=2, scale=1, unit=""):
    if value is None:
        return "N/A"
    return f"{value * scale:.{digits}f}".replace(".", ",") + unit


def load_training_summary(run, explicit_checkpoint=None):
    checkpoint = explicit_checkpoint or (
        Path(run["checkpoint"]) if run.get("checkpoint") else None
    )
    if not checkpoint or not (Path(checkpoint) / "training.json").is_file():
        return None
    info = json.loads((Path(checkpoint) / "training.json").read_text())
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "training_regime": info.get("training_regime"),
        "quantization": info.get("quantization", {"enabled": False, "dtype": "bfloat16"}),
        "optimizer": info.get("optimizer", "adamw"),
        "selected_epoch": info.get("selected_epoch"),
        "completed_epochs": info.get("completed_epochs"),
        "trainable_parameters": info.get("trainable_parameters"),
        "elapsed_seconds": info.get("elapsed_seconds"),
        "peak_allocated_vram_gib": info.get("peak_allocated_vram_gib"),
    }


def compare_qlora(
    lora_dir,
    qlora_dir,
    lora_checkpoint=None,
    qlora_checkpoint=None,
):
    lora_dir, qlora_dir = Path(lora_dir), Path(qlora_dir)
    models = {}
    for label, path, ckpt in [
        ("LoRA-BF16", lora_dir, lora_checkpoint),
        ("QLoRA-NF4", qlora_dir, qlora_checkpoint),
    ]:
        run = json.loads((path / "run.json").read_text())
        rows = read_jsonl(path / "predictions.jsonl")
        if run["status"] != "completed" or len(rows) != run["expected_records"]:
            raise ValueError(f"Avaliação incompleta: {label}")
        result = metrics(rows)
        counter = (
            "nvml_process"
            if "max_observed_nvml_process_vram_gib" in run
            else run.get("memory_kind", "dedicated_vram")
        )
        memory = (
            run["max_observed_nvml_process_vram_gib"]
            if counter == "nvml_process"
            else result.get("peak_allocated_vram_gib")
        )
        models[label] = {
            "run": run,
            "rows": rows,
            "metrics": result,
            "by_type": {
                kind: metrics([r for r in rows if r["type"] == kind]) for kind in TYPES
            },
            "in_domain": {
                kind: metrics(
                    [r for r in rows if not r["held_out_task"] and r["type"] == kind]
                )
                for kind in TYPES
            },
            "held_out_tasks": {
                kind: metrics(
                    [r for r in rows if r["held_out_task"] and r["type"] == kind]
                )
                for kind in TYPES
            },
            "by_task": {
                task: metrics([r for r in rows if r["task"] == task])
                for task in sorted({r["task"] for r in rows})
            },
            "memory_gib": memory,
            "memory_counter": counter,
            "training": load_training_summary(run, ckpt),
        }
    lora, qlora = models["LoRA-BF16"], models["QLoRA-NF4"]
    for key in ["gpu", "batch_size", "warmup_per_type", "capacity"]:
        if lora["run"].get(key) != qlora["run"].get(key):
            raise ValueError(f"Protocolo diferente: {key}")
    if lora["memory_counter"] != qlora["memory_counter"]:
        raise ValueError(
            "Contadores de memória diferentes; compare as duas avaliações no mesmo backend"
        )

    def test_hash(run):
        return run.get("source_sha256") or run["data"]["test"]["sha256"]

    if test_hash(lora["run"]) != test_hash(qlora["run"]):
        raise ValueError("Hashes do teste diferentes")
    fields = ["id", "task", "type", "held_out_task", "target", "option_ids"]
    a, b = lora["rows"], qlora["rows"]
    if len(a) != len(b) or any(any(x[k] != y[k] for k in fields) for x, y in zip(a, b)):
        raise ValueError("IDs, alvos ou alternativas diferentes")
    agreement = sum(
        x["prediction"] == y["prediction"]
        for x, y in zip(a, b)
        if x["error"] is None and y["error"] is None
    )
    valid_pairs = sum(
        x["error"] is None and y["error"] is None for x, y in zip(a, b)
    )
    report = {
        "same_cohort": True,
        "records": len(a),
        "valid_pairs": valid_pairs,
        "prediction_agreement_rate": agreement / valid_pairs if valid_pairs else None,
        "memory_counter": lora["memory_counter"],
        "models": {
            name: {k: v for k, v in m.items() if k != "rows"}
            for name, m in models.items()
        },
    }
    return report


def format_markdown(report):
    lora = report["models"]["LoRA-BF16"]
    qlora = report["models"]["QLoRA-NF4"]
    lines = [
        "# Comparação Pareada: DecisionLFM (LoRA BF16) vs DecisionLFM-QLoRA (4-bit NF4)",
        "",
        f"Mesmos {report['records']} exemplos de teste, batch 1, mesmo hardware (`{lora['run']['gpu']}`). "
        f"Concordância de previsão (`argmax`): {fmt(report['prediction_agreement_rate'], 2, 100, '%')}.",
        "",
        "## Métricas Globais no Teste",
        "",
        "| Métrica | LoRA (BF16) | QLoRA (4-bit NF4) |",
        "|---|---:|---:|",
    ]
    for label, key, scale, digits, unit in [
        ("Acurácia", "accuracy", 100, 2, "%"),
        ("Macro-F1", "macro_f1", 100, 2, "%"),
        ("Brier ↓", "brier_multiclass", 1, 6, ""),
        ("MAE de score ↓", "score_mae", 1, 6, ""),
        ("Latência mediana", "latency_median_ms", 1, 2, " ms"),
        ("Latência p95", "latency_p95_ms", 1, 2, " ms"),
        ("Falhas", "failed_records", 1, 0, ""),
        ("Tokens de entrada (mediana)", "input_tokens_median", 1, 0, ""),
    ]:
        lines.append(
            f"| {label} | {fmt(lora['metrics'].get(key), digits, scale, unit)} | "
            f"{fmt(qlora['metrics'].get(key), digits, scale, unit)} |"
        )
    lines.append(
        f"| Memória GPU inferência ({report['memory_counter']}) | "
        f"{fmt(lora['memory_gib'], 2, 1, ' GiB')} | {fmt(qlora['memory_gib'], 2, 1, ' GiB')} |"
    )
    if lora["training"] and qlora["training"]:
        lt, qt = lora["training"], qlora["training"]
        lines += [
            "",
            "## Custo e Recursos de Treinamento",
            "",
            "| Recurso | LoRA (BF16) | QLoRA (4-bit NF4) |",
            "|---|---:|---:|",
            f"| Pico de VRAM no treino | {fmt(lt.get('peak_allocated_vram_gib'), 2, 1, ' GiB')} | "
            f"{fmt(qt.get('peak_allocated_vram_gib'), 2, 1, ' GiB')} |",
            f"| Tempo total de treino | {fmt(lt.get('elapsed_seconds'), 1, 1, ' s')} | "
            f"{fmt(qt.get('elapsed_seconds'), 1, 1, ' s')} |",
            f"| Parâmetros treináveis | {lt.get('trainable_parameters', 'N/A')} | "
            f"{qt.get('trainable_parameters', 'N/A')} |",
            f"| Época selecionada | {lt.get('selected_epoch', 'N/A')} | "
            f"{qt.get('selected_epoch', 'N/A')} |",
        ]
    lines += [
        "",
        "## Desempenho por Tipo de Decisão",
        "",
        "| Tipo | Acurácia LoRA / QLoRA | Brier ↓ LoRA / QLoRA | MAE ↓ LoRA / QLoRA |",
        "|---|---:|---:|---:|",
    ]
    for kind in TYPES:
        lm, qm = lora["by_type"][kind], qlora["by_type"][kind]
        mae_cell = (
            f"{fmt(lm.get('score_mae'), 6)} / {fmt(qm.get('score_mae'), 6)}"
            if kind == "score"
            else "—"
        )
        lines.append(
            f"| `{kind}` | {fmt(lm['accuracy'], 2, 100, '%')} / {fmt(qm['accuracy'], 2, 100, '%')} | "
            f"{fmt(lm['brier_multiclass'], 6)} / {fmt(qm['brier_multiclass'], 6)} | {mae_cell} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora", type=Path, required=True, help="Diretório de avaliação do LoRA BF16")
    parser.add_argument("--qlora", type=Path, required=True, help="Diretório de avaliação do QLoRA 4-bit")
    parser.add_argument("--lora-checkpoint", type=Path, help="Opcional: checkpoint do LoRA BF16")
    parser.add_argument("--qlora-checkpoint", type=Path, help="Opcional: checkpoint do QLoRA 4-bit")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use um diretório novo")
    report = compare_qlora(
        args.lora,
        args.qlora,
        lora_checkpoint=args.lora_checkpoint,
        qlora_checkpoint=args.qlora_checkpoint,
    )
    markdown = format_markdown(report)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "comparison-qlora.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    (args.output_dir / "COMPARACAO_QLORA.md").write_text(markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()

