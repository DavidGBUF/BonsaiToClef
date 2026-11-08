"""Compara duas avaliações concluídas nos mesmos exemplos e hardware."""

import argparse
import json
from pathlib import Path

from dados_tipos import read_jsonl
from metricas import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clef", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use um diretório novo")
    models = {}
    for name, path in [("CLEF-Flash", args.clef), ("DecisionLFM", args.decision)]:
        run = json.loads((path / "run.json").read_text())
        rows = read_jsonl(path / "predictions.jsonl")
        if run["status"] != "completed" or len(rows) != run["expected_records"]:
            raise ValueError(f"Avaliação incompleta: {name}")
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
        models[name] = {
            "run": run,
            "rows": rows,
            "metrics": result,
            "memory_gib": memory,
            "memory_counter": counter,
        }
    clef, decision = models.values()
    for key in ["gpu", "batch_size", "warmup_per_type", "quantization", "capacity"]:
        if clef["run"].get(key) != decision["run"].get(key):
            raise ValueError(f"Protocolo diferente: {key}")
    if clef["memory_counter"] != decision["memory_counter"]:
        raise ValueError(
            "Contadores de memória diferentes; compare as duas avaliações no mesmo backend"
        )

    def test_hash(run):
        return run.get("source_sha256") or run["data"]["test"]["sha256"]

    if test_hash(clef["run"]) != test_hash(decision["run"]):
        raise ValueError("Hashes do teste diferentes")
    fields = ["id", "task", "type", "held_out_task", "target", "option_ids"]
    a, b = clef["rows"], decision["rows"]
    if len(a) != len(b) or any(any(x[k] != y[k] for k in fields) for x, y in zip(a, b)):
        raise ValueError("IDs, alvos ou alternativas diferentes")
    paired = {}
    upper = max(
        (max(x.get("input_tokens", 0), y.get("input_tokens", 0)) for x, y in zip(a, b)),
        default=8192,
    )
    for low, high in [
        (1, 256),
        (257, 512),
        (513, 1024),
        (1025, 2048),
        (2049, 8192),
        (8193, max(8193, upper)),
    ]:
        indices = [
            i
            for i, (x, y) in enumerate(zip(a, b))
            if x["error"] is None
            and y["error"] is None
            and low <= max(x["input_tokens"], y["input_tokens"]) <= high
        ]
        if indices:
            paired[f"{low}-{high}"] = {
                "ids": [a[i]["id"] for i in indices],
                **{
                    name: metrics([model["rows"][i] for i in indices])
                    for name, model in models.items()
                },
            }
    lines = [
        "# Comparação nos mesmos exemplos",
        "",
        "| Métrica | CLEF-Flash | DecisionLFM |",
        "|---|---:|---:|",
    ]
    for label, key, scale, digits, unit in [
        ("Acurácia", "accuracy", 100, 2, "%"),
        ("Macro-F1", "macro_f1", 100, 2, "%"),
        ("Brier ↓", "brier_multiclass", 1, 6, ""),
        ("Latência mediana", "latency_median_ms", 1, 2, " ms"),
        ("Latência p95", "latency_p95_ms", 1, 2, " ms"),
        ("Falhas", "failed_records", 1, 0, ""),
        ("Tokens de entrada: mediana", "input_tokens_median", 1, 0, ""),
        ("Tokens de entrada: p95", "input_tokens_p95", 1, 0, ""),
    ]:
        values = [
            f"{m['metrics'][key] * scale:.{digits}f}".replace(".", ",") + unit
            if m["metrics"].get(key) is not None
            else "N/A"
            for m in models.values()
        ]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    values = [
        f"{m['memory_gib']:.2f}".replace(".", ",") + " GiB"
        if m["memory_gib"] is not None
        else "N/A"
        for m in models.values()
    ]
    lines.append(
        f"| Memória GPU ({clef['memory_counter']}) | {values[0]} | {values[1]} |"
    )
    lines += [
        "",
        f"Mesmos {len(a)} registros, batch 1. Brier na escala [0,2]; Macro-F1 médio por tarefa.",
        "Memória NVML é o máximo observado do processo e inclui contexto CUDA; memória PyTorch é alocação de tensores.",
        "As faixas pareadas pelo maior prompt do par estão em comparison.json, com IDs e todas as métricas.",
    ]
    lines += [
        "",
        "## Faixas de tokens pareadas",
        "",
        "| Maior prompt do par | N | Tokens medianos CLEF / Decision | Acurácia CLEF / Decision | Brier CLEF / Decision | Mediana ms CLEF / Decision | p95 ms CLEF / Decision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, bucket in paired.items():
        cells = [label, str(len(bucket["ids"]))]
        for key, scale, digits in [
            ("input_tokens_median", 1, 0),
            ("accuracy", 100, 2),
            ("brier_multiclass", 1, 6),
            ("latency_median_ms", 1, 2),
            ("latency_p95_ms", 1, 2),
        ]:
            cells.append(
                " / ".join(
                    f"{bucket[name][key] * scale:.{digits}f}".replace(".", ",")
                    for name in models
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    report = {
        "same_cohort": True,
        "records": len(a),
        "memory_counter": clef["memory_counter"],
        "models": {
            name: {k: v for k, v in m.items() if k != "rows"}
            for name, m in models.items()
        },
        "paired_token_buckets": paired,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output_dir / "COMPARACAO.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

