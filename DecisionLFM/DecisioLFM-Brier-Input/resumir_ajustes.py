"""Audita busca ordenada, calibração com proteção de MAE e teste pareado."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from calibracao import choose_candidate, load_calibration, summarize_logits, weight_hash
from comparar_ajustes import CONFIGURATIONS
from dados_tipos import digest, read_jsonl
from metricas import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    info = json.loads((run / "experiment.json").read_text())
    root = Path(__file__).resolve().parent
    for relative, sha in info["code_sha256"].items():
        assert digest(root / relative) == sha, relative
    data = Path(info["data_dir"])
    manifest = json.loads((data / "manifest.json").read_text())
    assert digest(data / "manifest.json") == info["data_manifest_sha256"]
    for split in ["train", "validation", "test"]:
        assert digest(data / f"{split}.jsonl") == manifest[split]["sha256"]
    expected = read_jsonl(data / "test.jsonl")
    validation = {r["id"]: r for r in read_jsonl(data / "validation.jsonl")}
    configurations = {c["name"]: c for c in CONFIGURATIONS}
    partitions = None
    for model in info["models"]:
        checkpoint = Path(model["checkpoint"])
        training = json.loads((checkpoint / "training.json").read_text())
        adapter = json.loads((checkpoint / "adapter/adapter_config.json").read_text())
        assert (
            training["status"] == "completed"
            and adapter["r"] == 8
            and adapter["lora_alpha"] == 16
        )
        replay = json.loads((run / f"replay-{model['name']}.json").read_text())
        assert (
            replay["status"] == "passed"
            and replay["records"] == manifest["validation"]["records"]
        )
        assert replay["max_abs_loss_difference"] <= 1e-5
        calibration = json.loads((checkpoint / "calibration.json").read_text())
        temperatures = load_calibration(checkpoint, verify_weights=True)
        assert calibration["score_mae_protected"]
        fit = calibration["by_type"]["score"]
        assert fit["mae_after"] <= fit["mae_before"] + 1e-12
        assert fit["after"] <= fit["before"] + 1e-12
        left, right = (
            set(calibration["selection_ids"]),
            set(calibration["calibration_ids"]),
        )
        assert not left & right and left | right == set(validation)
        assert partitions is None or partitions == (left, right)
        partitions = (left, right)
        assert not {validation[k]["group"] for k in left} & {
            validation[k]["group"] for k in right
        }
        assert set(Counter(validation[k]["type"] for k in left)) == {
            "choice",
            "noul",
            "score",
        }
        cache = read_jsonl(checkpoint / "validation-logits.jsonl")
        assert {r["id"] for r in cache} == set(validation)
        selected_rows = [r for r in cache if r["id"] in left]
        for mode, ts, key in [
            ("raw", {}, "selection_before"),
            ("protected", temperatures, "selection_after"),
        ]:
            actual = summarize_logits(selected_rows, ts)
            assert actual == calibration[key]
            variant = next(
                c for c in info["variants"] if c["name"] == f"{model['name']}-{mode}"
            )
            assert variant["validation"] == actual
        if model["source_checkpoint"]:
            assert weight_hash(checkpoint) == weight_hash(
                Path(model["source_checkpoint"])
            )
        else:
            config = configurations[model["name"]]
            assert [h["epoch"] for h in training["history"]] == list(
                range(1, len(training["history"]) + 1)
            )
            assert 1 <= len(training["history"]) <= 3
            assert all(
                h["optimizer_steps"] == math.ceil(manifest["train"]["records"] / 16)
                for h in training["history"]
            )
            assert (
                training["args"]["selection_metric"] == "accuracy"
                and training["args"]["lr_scheduler"] == "constant"
            )
            for key in ["head_lr", "lora_lr", "score_brier_weight"]:
                assert training["args"][key] == config[key]
            assert training["objective"]["brier_weight"] == 1
            assert all(
                h["learning_rates"]
                == h["next_learning_rates"]
                == [config["head_lr"], config["lora_lr"]]
                for h in training["history"]
            )
            best = max(
                training["history"],
                key=lambda h: (
                    h["validation"]["selection"]["mean_type_accuracy"],
                    -h["validation"]["selection"]["mean_type_brier"],
                ),
            )
            assert best["epoch"] == training["selected_epoch"]
    baseline = next(
        v["validation"] for v in info["variants"] if v["name"] == "original-raw"
    )
    assert choose_candidate(info["variants"], baseline) == info["selected"]
    final = Path(info["final_checkpoint"])
    assert weight_hash(final) == weight_hash(Path(info["selected"]["checkpoint"]))
    if info["selected"]["calibrated"]:
        assert load_calibration(final, True)
    else:
        assert not (final / "calibration.json").exists()

    def audit_test(directory):
        metadata = json.loads((directory / "run.json").read_text())
        rows = read_jsonl(directory / "predictions.jsonl")
        assert metadata["status"] == "completed" and len(rows) == len(expected)
        assert (
            metadata.get("source_sha256") or metadata["data"]["test"]["sha256"]
        ) == info["test_sha256"]
        assert not any(r["error"] for r in rows)
        for row, source in zip(rows, expected):
            assert row["id"] == source["id"] and row["target"] == str(
                next(iter(source["targets"].values()))
            )
            assert all(
                math.isfinite(p) and 0 <= p <= 1 for p in row["probabilities"].values()
            )
            assert math.isclose(sum(row["probabilities"].values()), 1, abs_tol=1e-5)
        result = metrics(rows)
        brier = math.fsum(
            math.fsum(
                (p - (k == row["target"])) ** 2 for k, p in row["probabilities"].items()
            )
            for row in rows
        ) / len(rows)
        assert math.isclose(brier, result["brier_multiclass"], abs_tol=1e-12)
        memory = metadata.get(
            "max_observed_nvml_process_vram_gib", result["peak_allocated_vram_gib"]
        )
        return {
            "metrics": result,
            "memory_gib": memory,
            "run": metadata,
            "by_type": {
                kind: metrics([r for r in rows if r["type"] == kind])
                for kind in ["choice", "noul", "score"]
            },
        }, rows

    result = {
        "status": "completed",
        "records": len(expected),
        "selected": info["selected"],
        "models": info["models"],
        "variants": info["variants"],
        "native": {},
        "gguf": {},
    }
    source_rows = None
    for variant in info["variants"]:
        measured, rows = audit_test(run / variant["name"])
        result["native"][variant["name"]] = measured
        if source_rows is not None:
            assert all(
                all(
                    a[k] == b[k]
                    for k in [
                        "id",
                        "task",
                        "type",
                        "target",
                        "option_ids",
                        "input_tokens",
                    ]
                )
                for a, b in zip(source_rows, rows)
            )
        source_rows = rows
    for model in info["models"]:
        a = read_jsonl(run / f"{model['name']}-raw/predictions.jsonl")
        b = read_jsonl(run / f"{model['name']}-protected/predictions.jsonl")
        assert all(x["prediction"] == y["prediction"] for x, y in zip(a, b))
    if info.get("export"):
        reference_tokens = None
        for quant in ["f16", "q8_0", "q4_0"]:
            pair, rows_pair = {}, []
            for mode in ["raw", "protected"]:
                measured, rows = audit_test(run / f"gguf-{quant}-{mode}")
                pair[mode] = measured
                rows_pair.append(rows)
            assert all(
                x["prediction"] == y["prediction"]
                and x["input_sha256"] == y["input_sha256"]
                for x, y in zip(*rows_pair)
            )
            if reference_tokens is not None:
                assert [r["input_sha256"] for r in rows_pair[0]] == reference_tokens
            reference_tokens = [r["input_sha256"] for r in rows_pair[0]]
            result["gguf"][quant] = pair
    hashes = {
        str(p.relative_to(final)): digest(p)
        for p in sorted(final.rglob("*"))
        if p.is_file()
    }
    (run / "checkpoint-final-sha256.json").write_text(
        json.dumps(hashes, indent=2) + "\n"
    )
    (run / "comparison-ordered.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )

    def fmt(value, digits=2, scale=1):
        return f"{value * scale:.{digits}f}".replace(".", ",")

    lines = [
        "# Ajustes ordenados: acurácia e proteção do MAE de score",
        "",
        "Seleção por validação: acurácia e MAE de score não piores que o original sem temperatura; maior acurácia, com Brier como desempate. A proteção de MAE no ajuste de T é uma restrição na validação, não uma garantia sobre o teste.",
        "",
        f"Selecionado antes do teste: `{info['selected']['name']}`.",
        "",
        "## Validação de seleção",
        "",
        "| Variante | Acurácia | Brier ↓ | MAE score ↓ |",
        "|---|---:|---:|---:|",
    ]
    for variant in info["variants"]:
        v = variant["validation"]
        lines.append(
            f"| {variant['name']} | {fmt(v['mean_type_accuracy'], scale=100)}% | {fmt(v['mean_type_brier'], 6)} | {fmt(v['types']['score']['mae'], 6)} |"
        )
    lines += [
        "",
        "## Teste PyTorch BF16",
        "",
        "| Variante | Acurácia | Macro-F1 | Brier ↓ | p50 ms | p95 ms | VRAM GiB | Falhas | MAE score ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, measured in result["native"].items():
        m = measured["metrics"]
        lines.append(
            f"| {name} | {fmt(m['accuracy'], scale=100)}% | {fmt(m['macro_f1'], scale=100)}% | {fmt(m['brier_multiclass'], 6)} | {fmt(m['latency_median_ms'])} | {fmt(m['latency_p95_ms'])} | {fmt(measured['memory_gib'])} | {m['failed_records']} | {fmt(m['score_mae'], 6)} |"
        )
    for quant, pair in result["gguf"].items():
        lines += [
            "",
            f"## GGUF {quant.upper()}: pesos selecionados",
            "",
            "| Métrica | T=1 | Temperatura selecionada |",
            "|---|---:|---:|",
        ]
        for label, key, digits, scale in [
            ("Acurácia %", "accuracy", 2, 100),
            ("Macro-F1 %", "macro_f1", 2, 100),
            ("Brier ↓", "brier_multiclass", 6, 1),
            ("Latência mediana ms", "latency_median_ms", 2, 1),
            ("Latência p95 ms", "latency_p95_ms", 2, 1),
            ("Falhas", "failed_records", 0, 1),
            ("MAE score ↓", "score_mae", 6, 1),
        ]:
            lines.append(
                f"| {label} | {fmt(pair['raw']['metrics'][key], digits, scale)} | {fmt(pair['protected']['metrics'][key], digits, scale)} |"
            )
        lines.append(
            f"| VRAM NVML GiB | {fmt(pair['raw']['memory_gib'])} | {fmt(pair['protected']['memory_gib'])} |"
        )
    lines += [
        "",
        "PyTorch: pico alocado de tensores. GGUF: máximo observado NVML incluindo contexto CUDA. Uma medição por configuração; pequenas diferenças de latência não isolam causalidade. Brier soma classes [0,2]; Macro-F1 agregado é a média por tarefa. Todos os ajustes foram fixados antes de ler o teste.",
        "",
    ]
    (run / "COMPARACAO_AJUSTES.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
