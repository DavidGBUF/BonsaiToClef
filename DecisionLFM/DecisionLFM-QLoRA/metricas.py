"""Qualidade, latência, memória e tokens no mesmo protocolo dos experimentos."""

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def metrics(rows):
    if not rows:
        return {
            "records": 0,
            "failed_records": 0,
            "failure_rate": None,
            "accuracy": None,
            "macro_f1": None,
            "brier_multiclass": None,
            "probability_metrics_records": 0,
            "latency_median_ms": None,
            "latency_p95_ms": None,
            "peak_allocated_vram_gib": None,
        }
    valid = [row for row in rows if row["error"] is None]
    task_metrics = []
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    for items in by_task.values():
        labels = sorted({key for row in items for key in row["option_ids"]})
        task_metrics.append(
            float(
                f1_score(
                    [r["target"] for r in items],
                    [r["prediction"] or "__error__" for r in items],
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            )
        )
    brier = [
        sum(
            (p - int(key == row["target"])) ** 2
            for key, p in row["probabilities"].items()
        )
        for row in valid
    ]
    latency = [row["latency_ms"] for row in valid]
    result = {
        "records": len(rows),
        "failed_records": len(rows) - len(valid),
        "failure_rate": (len(rows) - len(valid)) / len(rows),
        "accuracy": float(
            accuracy_score(
                [r["target"] for r in rows],
                [r["prediction"] or "__error__" for r in rows],
            )
        ),
        "macro_f1": float(np.mean(task_metrics)),
        "brier_multiclass": float(np.mean(brier)) if valid else None,
        "probability_metrics_records": len(valid),
        "latency_median_ms": float(np.median(latency)) if valid else None,
        "latency_p95_ms": float(np.percentile(latency, 95)) if valid else None,
        "peak_allocated_vram_gib": max(
            (
                r["peak_allocated_vram_gib"]
                for r in valid
                if r["peak_allocated_vram_gib"] is not None
            ),
            default=None,
        ),
    }
    if any("mps_allocated_gib" in row for row in valid):
        result["max_observed_mps_allocated_gib"] = max(
            row["mps_allocated_gib"] for row in valid
        )
    scores = [
        abs(row["score"] - int(row["target"]))
        for row in valid
        if row["type"] == "score"
    ]
    if scores:
        result["score_mae"] = float(np.mean(scores))
    if valid and all("input_tokens" in row for row in valid):
        tokens = [row["input_tokens"] for row in valid]
        result.update(
            input_tokens_mean=float(np.mean(tokens)),
            input_tokens_median=float(np.median(tokens)),
            input_tokens_p95=float(np.percentile(tokens, 95)),
            input_tokens_max=max(tokens),
        )
    return result

