"""Audita teste pareado antes/depois da temperatura e escreve as métricas."""

import argparse
import json
import math
from pathlib import Path

from calibracao import TYPES, load_calibration
from dados_tipos import digest, read_jsonl
from metricas import metrics


def audit_rows(rows):
    valid = [r for r in rows if r["error"] is None]
    for row in valid:
        ps = row["probabilities"]
        if set(ps) != set(row["option_ids"]) or any(
            not math.isfinite(p) or p < 0 or p > 1 for p in ps.values()
        ):
            raise ValueError("Distribuição inválida")
        if not math.isclose(sum(ps.values()), 1, abs_tol=1e-5):
            raise ValueError("Probabilidades não somam um")
    result = metrics(rows)
    brier = math.fsum(
        math.fsum((p - (k == r["target"])) ** 2 for k, p in r["probabilities"].items())
        for r in valid
    ) / len(valid)
    if not math.isclose(brier, result["brier_multiclass"], abs_tol=1e-12):
        raise ValueError("Brier recalculado divergiu")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    info = json.loads((a.run / "experiment.json").read_text())
    root = Path(__file__).resolve().parent
    for relative, sha in info["code_sha256"].items():
        if digest(root / relative) != sha:
            raise ValueError(f"Código alterado durante a execução: {relative}")
    selected = info["selected"]
    checkpoint = Path(selected["checkpoint"])
    temperatures = load_calibration(checkpoint, verify_weights=True)
    calibration = json.loads((checkpoint / "calibration.json").read_text())
    selection_ids, calibration_ids = (
        set(calibration["selection_ids"]),
        set(calibration["calibration_ids"]),
    )
    if selection_ids & calibration_ids:
        raise ValueError("Partições de validação sobrepostas")
    if selected != min(info["candidates"], key=lambda c: c["selection_brier"]):
        raise ValueError("Escolha divergiu da regra prévia de validação")
    for candidate in info["candidates"]:
        training = json.loads(
            (Path(candidate["checkpoint"]) / "training.json").read_text()
        )
        replay = json.loads((a.run / f"replay-{candidate['name']}.json").read_text())
        if not (
            training["status"] == "completed"
            and training["args"]["lora_rank"] == 8
            and len(training["history"]) <= 3
            and replay["status"] == "passed"
        ):
            raise ValueError(f"Candidato incompleto: {candidate['name']}")
        if set(training["validation_partition"]["selection"]) != selection_ids:
            raise ValueError("Seleção usa IDs diferentes")
        if set(training["validation_partition"]["calibration"]) != calibration_ids:
            raise ValueError("Calibração usa IDs diferentes")
    data = Path(info["data_dir"])
    manifest = json.loads((data / "manifest.json").read_text())
    if (
        digest(data / "test.jsonl") != info["test_sha256"]
        or digest(data / "manifest.json") != info["data_manifest_sha256"]
    ):
        raise ValueError("Dados alterados durante a execução")
    expected = read_jsonl(data / "test.jsonl")
    if len(expected) != manifest["test"]["records"]:
        raise ValueError("Contagem de teste diferente do manifesto")
    test_ids = {r["id"] for r in expected}
    if test_ids & (selection_ids | calibration_ids):
        raise ValueError("Teste presente na validação")
    formats = ["pytorch"] + (
        ["gguf-f16", "gguf-q8_0", "gguf-q4_0"] if info.get("export") else []
    )
    historical = (
        json.loads(Path(info["historical"]).read_text())
        if info.get("historical")
        else None
    )
    if historical and digest(Path(info["historical"])) != info["historical_sha256"]:
        raise ValueError("Controle histórico alterado")
    if historical and historical["experiment"]["test_sha256"] != info["test_sha256"]:
        raise ValueError("Teste histórico diferente")
    result = {
        "status": "completed",
        "records": len(expected),
        "selected": selected,
        "temperatures": temperatures,
        "calibration_by_type": calibration["by_type"],
        "formats": {},
    }
    lines = [
        "# DecisionLFM-QLoRA: até três épocas, seleção por Brier e temperatura por tipo",
        "",
        "Todos os ajustes usam somente validação: metade para época/configuração e metade para temperatura. "
        "As colunas sem/com calibração usam exatamente os mesmos pesos. Teste executado depois da seleção.",
        "",
        f"Configuração: `{selected['name']}`; época selecionada: {selected['selected_epoch']}; QLoRA 4-bit NF4 rank 8.",
        "",
        "Temperaturas: "
        + ", ".join(f"{k}={v:.6f}" for k, v in temperatures.items())
        + ".",
        "",
    ]
    reference = None
    for name in formats:
        pair, rows_pair = {}, []
        for mode in ["uncalibrated", "calibrated"]:
            directory = a.run / f"{name}-{mode}"
            metadata = json.loads((directory / "run.json").read_text())
            rows = read_jsonl(directory / "predictions.jsonl")
            sha = metadata.get("source_sha256") or metadata["data"]["test"]["sha256"]
            if (
                metadata["status"] != "completed"
                or len(rows) != len(expected)
                or sha != info["test_sha256"]
            ):
                raise ValueError(f"Teste incompleto: {directory}")
            for row, sample in zip(rows, expected):
                key = next(iter(sample["targets"]))
                if row["id"] != sample["id"] or row["target"] != str(
                    sample["targets"][key]
                ):
                    raise ValueError("IDs/alvos diferentes do teste")
            m = audit_rows(rows)
            memory = (
                m["peak_allocated_vram_gib"]
                if name == "pytorch"
                else metadata["max_observed_nvml_process_vram_gib"]
            )
            pair[mode] = {
                "metrics": m,
                "memory_gib": memory,
                "by_type": {
                    kind: metrics([r for r in rows if r["type"] == kind])
                    for kind in TYPES
                },
                "by_task": {
                    task: metrics([r for r in rows if r["task"] == task])
                    for task in sorted({r["task"] for r in rows})
                },
                "run": metadata,
            }
            rows_pair.append(rows)
        for x, y in zip(*rows_pair):
            if any(
                x[k] != y[k] for k in ["id", "task", "type", "target", "option_ids"]
            ):
                raise ValueError("Teste pareado divergiu")
            if x["error"] is None and y["error"] is None:
                if (
                    x["input_tokens"] != y["input_tokens"]
                    or x["prediction"] != y["prediction"]
                ):
                    raise ValueError("Temperatura alterou tokens ou argmax")
                if name != "pytorch" and x["input_sha256"] != y["input_sha256"]:
                    raise ValueError("IDs de tokens alterados")
        if name != "pytorch":
            if reference is not None:
                if any(
                    x.get("input_sha256") != y.get("input_sha256")
                    for x, y in zip(reference, rows_pair[1])
                ):
                    raise ValueError("Tokenização diferente entre formatos GGUF")
            reference = rows_pair[1]
        old_name = "pytorch-bf16" if name == "pytorch" else name
        if historical:
            pair["historical_3ep"] = historical["models"][old_name]["3"]
            old = pair["historical_3ep"]
            if old.get("predictions_path"):
                previous = read_jsonl(Path(old["predictions_path"]))
                if len(previous) != len(rows_pair[0]):
                    raise ValueError("Coorte histórica diferente")
                for x, y in zip(previous, rows_pair[0]):
                    if any(
                        x[k] != y[k]
                        for k in ["id", "task", "type", "target", "option_ids"]
                    ):
                        raise ValueError("IDs/alvos históricos diferentes")
                    if x["error"] is None and y["error"] is None:
                        if x["input_tokens"] != y["input_tokens"]:
                            raise ValueError("Número de tokens histórico diferente")
                        if name != "pytorch" and x["input_sha256"] != y["input_sha256"]:
                            raise ValueError("IDs dos tokens históricos diferentes")
        result["formats"][name] = pair
        lines += [
            f"## {name}",
            "",
            "| Métrica | Sem calibração | Com calibração |",
            "|---|---:|---:|",
        ]
        for label, key, scale, digits, unit in [
            ("Acurácia", "accuracy", 100, 2, "%"),
            ("Macro-F1", "macro_f1", 100, 2, "%"),
            ("Brier ↓", "brier_multiclass", 1, 6, ""),
            ("Latência mediana", "latency_median_ms", 1, 2, " ms"),
            ("Latência p95", "latency_p95_ms", 1, 2, " ms"),
            ("Falhas", "failed_records", 1, 0, ""),
        ]:
            values = [
                f"{pair[m]['metrics'][key] * scale:.{digits}f}".replace(".", ",") + unit
                for m in ["uncalibrated", "calibrated"]
            ]
            lines.append(f"| {label} | {' | '.join(values)} |")
        values = [
            f"{pair[m]['memory_gib']:.2f}".replace(".", ",")
            for m in ["uncalibrated", "calibrated"]
        ]
        lines += [f"| Memória GPU | {values[0]} GiB | {values[1]} GiB |", ""]
    lines += [
        "PyTorch: pico alocado de tensores. GGUF: máximo observado NVML, incluindo contexto CUDA. "
        "Os contadores não são equivalentes. Brier usa soma de classes [0,2]; Macro-F1 agregado é a média por tarefa. "
        "As métricas de score incluem MAE no JSON. Temperaturas são ajustadas no PyTorch e transferidas ao GGUF, sem ajuste no teste.",
        "",
        "Comparações por tipo/tarefa e os controles históricos de três épocas estão em comparison-calibracao.json.",
        "",
    ]
    (a.run / "comparison-calibracao.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (a.run / "COMPARACAO_CALIBRACAO.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

