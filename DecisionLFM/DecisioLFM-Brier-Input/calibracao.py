"""Temperatura por tipo, ajustada somente em uma parte reservada da validação."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

TYPES = ("choice", "noul", "score")


def check_temperature(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Temperatura deve ser finita e > 0")
    return value


def probabilities(logits, temperature=1.0):
    return (logits.float() / check_temperature(temperature)).softmax(-1)


def weight_hash(directory):
    """Vincula a calibração aos pesos, sem incluir o próprio calibration.json."""
    directory = Path(directory)
    result = hashlib.sha256()
    files = [directory / "head.safetensors"]
    files += sorted((directory / "adapter").glob("*.safetensors"))
    files += [
        p
        for p in [
            directory / "decision_config.json",
            directory / "adapter/adapter_config.json",
        ]
        if p.is_file()
    ]
    for path in files:
        result.update(str(path.relative_to(directory)).encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def load_calibration(directory, verify_weights=False):
    directory = Path(directory)
    path = directory / "calibration.json"
    if not path.exists():
        return {}
    info = json.loads(path.read_text())
    if info["status"] != "completed" or info["method"] != "temperature_per_type":
        raise ValueError("Calibração incompleta ou desconhecida")
    if verify_weights and info["weight_hash"] != weight_hash(directory):
        raise ValueError("Calibração pertence a outros pesos")
    values = info["temperatures"]
    if set(values) != set(TYPES):
        raise ValueError("Calibração deve conter os três tipos")
    return {kind: check_temperature(values[kind]) for kind in TYPES}


def validation_partition(rows, seed=42):
    """Divide documentos por tarefa; o teste não participa de nenhum ajuste."""
    tasks = defaultdict(set)
    for row in rows:
        tasks[row["task"]].add(row["group"])
    selected, calibration = set(), set()
    for task, groups in sorted(tasks.items()):
        groups = sorted(
            groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).digest()
        )
        if len(groups) < 2:
            raise ValueError(f"Validação precisa de >= 2 documentos por tarefa: {task}")
        midpoint = len(groups) // 2
        selected.update(groups[:midpoint])
        calibration.update(groups[midpoint:])
    if selected & calibration:
        raise ValueError("Documento compartilhado entre partições de validação")
    result = {
        "selection": [r["id"] for r in rows if r["group"] in selected],
        "calibration": [r["id"] for r in rows if r["group"] in calibration],
    }
    for ids in result.values():
        scope = set(ids)
        if {r["type"] for r in rows if r["id"] in scope} != set(TYPES):
            raise ValueError("Cada partição de validação precisa dos três tipos")
    return result


def fit_temperature(logits, targets, protect_mae=False):
    """Busca escalar limitada [0,25; 16], incluindo T=1 como controle."""
    logits = logits.double().cpu()
    targets = targets.long().cpu()
    truth = F.one_hot(targets, logits.shape[-1]).double()

    def evaluate(t):
        ps = (logits / t).softmax(-1)
        brier = float((ps - truth).square().sum(-1).mean())
        expected = (ps * torch.arange(logits.shape[-1])).sum(-1)
        mae = float((expected - targets).abs().mean())
        return brier, mae

    baseline, baseline_mae = evaluate(1.0)
    best, value, best_mae = 1.0, baseline, baseline_mae
    low, high = math.log(0.25), math.log(16.0)
    for _ in range(2):
        grid = torch.linspace(low, high, 65).exp().tolist()
        for temperature in grid:
            candidate, mae = evaluate(temperature)
            if protect_mae and mae > baseline_mae + 1e-12:
                continue
            if candidate < value - 1e-12:
                best, value, best_mae = temperature, candidate, mae
        width = (high - low) / 64
        low = max(math.log(0.25), math.log(best) - width)
        high = min(math.log(16.0), math.log(best) + width)
    return best, {
        "before": baseline,
        "after": value,
        "records": len(targets),
        "mae_before": baseline_mae,
        "mae_after": best_mae,
        "mae_protected": protect_mae,
    }


def fit_by_type(rows, protect_score_mae=False):
    temperatures, metrics = {}, {}
    for kind in TYPES:
        items = [r for r in rows if r["type"] == kind]
        if not items:
            raise ValueError(f"Calibração sem exemplos de {kind}")
        width = max(len(r["logits"]) for r in items)
        logits = torch.full((len(items), width), -torch.inf, dtype=torch.float64)
        for i, row in enumerate(items):
            logits[i, : len(row["logits"])] = torch.tensor(
                row["logits"], dtype=torch.float64
            )
        targets = torch.tensor([r["target"] for r in items])
        temperatures[kind], metrics[kind] = fit_temperature(
            logits, targets, protect_mae=protect_score_mae and kind == "score"
        )
        if kind != "score":
            metrics[kind].pop("mae_before")
            metrics[kind].pop("mae_after")
    return temperatures, metrics


def summarize_logits(rows, temperatures=None):
    """Métricas de validação; os índices de score são os níveis ordinais."""
    temperatures = temperatures or {}
    types = {}
    for kind in TYPES:
        items = [r for r in rows if r["type"] == kind]
        if not items:
            raise ValueError(f"Validação sem {kind}")
        correct, brier, mae = 0, 0.0, 0.0
        for row in items:
            logits = torch.tensor(row["logits"], dtype=torch.float64)
            ps = (logits / check_temperature(temperatures.get(kind, 1.0))).softmax(-1)
            target = row["target"]
            correct += int(ps.argmax().item() == target)
            truth = F.one_hot(torch.tensor(target), len(ps)).double()
            brier += float((ps - truth).square().sum())
            mae += abs(float((ps * torch.arange(len(ps))).sum()) - target)
        types[kind] = {
            "records": len(items),
            "correct": correct,
            "accuracy": correct / len(items),
            "brier": brier / len(items),
        }
        if kind == "score":
            types[kind]["mae"] = mae / len(items)
    return {
        "types": types,
        "mean_type_accuracy": sum(v["accuracy"] for v in types.values()) / 3,
        "mean_type_brier": sum(v["brier"] for v in types.values()) / 3,
    }


def choose_candidate(candidates, reference):
    """Prioriza acurácia, com limites de acurácia/MAE definidos na validação."""
    accuracy = reference["mean_type_accuracy"]
    mae = reference["types"]["score"]["mae"]
    eligible = [
        c
        for c in candidates
        if c["validation"]["mean_type_accuracy"] >= accuracy - 1e-12
        and c["validation"]["types"]["score"]["mae"] <= mae + 1e-6
    ]
    if not eligible:
        raise ValueError("Nenhum candidato atende aos limites da validação")
    return min(
        eligible,
        key=lambda c: (
            -c["validation"]["mean_type_accuracy"],
            c["validation"]["mean_type_brier"],
            c["validation"]["types"]["score"]["mae"],
        ),
    )
