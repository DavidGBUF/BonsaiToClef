"""Avalia DecisionLFM ou CLEF GGUF: mesmos registros e protocolo do benchmark."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from runtime import Runtime, MemorySampler, gpu_name, LLAMA_COMMIT

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from esquema import prepare_record
from dados_tipos import read_jsonl, digest
from metricas import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=["clef", "decision"], required=True)
    p.add_argument("--backbone", type=Path, required=True)
    p.add_argument("--head", type=Path)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split", choices=["test", "validation"], default="test")
    p.add_argument("--limit-per-task", type=int)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quantization", choices=["F16", "Q8_0", "Q4_0"], required=True)
    p.add_argument("--library", type=Path)
    p.add_argument("--clef-dir", type=Path)
    p.add_argument("--capacity", type=int, default=2048)
    p.add_argument("--uncalibrated", action="store_true")
    a = p.parse_args()
    if a.output.exists() or a.capacity < 1:
        p.error("Use diretório novo e capacity > 0")
    if a.family == "decision" and (a.head is None or a.tokenizer is None):
        p.error("DecisionLFM requer --head e --tokenizer")
    torch.set_num_threads(8)
    manifest = json.loads((a.data_dir / "manifest.json").read_text())
    sha = digest(a.data_dir / f"{a.split}.jsonl")
    assert sha == manifest[a.split]["sha256"]
    samples = read_jsonl(a.data_dir / f"{a.split}.jsonl")
    if a.limit_per_task:
        counts = Counter()
        selected = []
        for sample in samples:
            if counts[sample["task"]] < a.limit_per_task:
                selected.append(sample)
                counts[sample["task"]] += 1
        samples = selected
    if not samples or any(
        len(s["questions"]) != 1 or len(s["targets"]) != 1 for s in samples
    ):
        raise ValueError("O benchmark requer uma pergunta e um alvo por registro")
    a.output.mkdir(parents=True)
    run = {
        "status": "running",
        "family": a.family,
        "backend": "llama.cpp + GGML",
        "quantization": a.quantization,
        "llama_commit": LLAMA_COMMIT,
        "gpu": gpu_name(),
        "library_sha256": digest(
            a.library if a.library else Path(__file__).with_name("libdecisionlfm.so")
        ),
        "llama_patch_sha256": digest(
            Path(__file__).with_name("llama-embedding-only.patch")
        ),
        "source_sha256": sha,
        "data": manifest,
        "split": a.split,
        "expected_records": len(samples),
        "limit_per_task": a.limit_per_task,
        "batch_size": 1,
        "warmup_per_type": 3,
        "threads": 8,
        "capacity": a.capacity,
        "max_length": a.capacity,
        "backbone": str(a.backbone),
        "head": str(a.head) if a.head else "native CLEF GGUF head",
        "memory_protocol": "GGML allocated CUDA buffers (weights+context+compute), sampled after synchronized calls. "
        "Also process VRAM sampled by NVML every 2 ms, includes CUDA context. Neither is Torch allocation.",
        "protocol": "Same original HF token IDs and option/question spans; no truncation, generation or prompt cache reuse. "
        "Latency includes schema normalization, tokenization, inference, CPU/GPU transfers, formatting, checks and synchronization; "
        "excludes loading, warmup and disk. Brier=sum over options [0,2]; macro-F1=unweighted mean across tasks. "
        "DecisionLFM includes GGUF query/option matrices executed on GGML CPU; backbone CUDA. "
        "CLEF complete backbone and head use native GGUF CUDA graph.",
        "code_sha256": {
            f.name: digest(f)
            for f in Path(__file__).parent.glob("*")
            if f.suffix in [".py", ".cpp"]
        },
    }
    out = a.output / "run.json"
    out.write_text(json.dumps(run, indent=2) + "\n")
    runtime = None
    rows = []
    try:
        runtime = Runtime(
            a.family,
            a.backbone,
            a.head,
            a.tokenizer,
            a.capacity,
            a.library,
            a.clef_dir,
            calibrated=not a.uncalibrated,
        )
        run["temperatures"] = runtime.temperatures
        for kind in ["choice", "noul", "score"]:
            first = next((s for s in samples if s["type"] == kind), None)
            if first is None:
                continue
            for _ in range(3):
                runtime.respond(first)
        with MemorySampler() as memory, (a.output / "predictions.jsonl").open("w") as f:
            for i, sample in enumerate(samples):
                _, questions = prepare_record(sample)
                key, q = next(iter(questions.items()))
                row = {k: sample[k] for k in ["id", "task", "type", "held_out_task"]}
                row.update(
                    target=str(sample["targets"][key]),
                    option_ids=list(q["options"]),
                    prediction=None,
                    probabilities=None,
                    error=None,
                    latency_ms=None,
                    peak_allocated_vram_gib=None,
                )
                try:
                    runtime.sync()
                    start = time.perf_counter()
                    response = runtime.respond(sample)
                    answer = response["answers"][key]
                    ps = (
                        {"true": answer["noul"], "false": 1 - answer["noul"]}
                        if sample["type"] == "noul"
                        else answer["probabilities"]
                    )
                    if (
                        set(ps) != set(row["option_ids"])
                        or not all(np.isfinite(v) and 0 <= v <= 1 for v in ps.values())
                        or not np.isclose(sum(ps.values()), 1, atol=1e-5)
                    ):
                        raise ValueError("Invalid probability distribution")
                    runtime.sync()
                    elapsed = (time.perf_counter() - start) * 1000
                    row.update(
                        prediction=max(ps, key=ps.get),
                        probabilities=ps,
                        latency_ms=elapsed,
                        input_tokens=response["usage"]["input_tokens"],
                        input_sha256=response["usage"]["input_sha256"],
                        peak_allocated_vram_gib=runtime.lib.df_vram(runtime.ptr)
                        / 1024**3,
                    )
                    if sample["type"] == "score":
                        row["score"] = answer["score"]
                except (RuntimeError, ValueError) as ex:
                    row["error"] = f"{type(ex).__name__}: {ex}"
                rows.append(row)
                f.write(json.dumps(row, allow_nan=False) + "\n")
                f.flush()
                if (i + 1) % 100 == 0 or i + 1 == len(samples):
                    print(
                        f"test={i + 1}/{len(samples)} failures={sum(r['error'] is not None for r in rows)}",
                        flush=True,
                    )
        run["max_observed_nvml_process_vram_gib"] = memory.peak / 1024**3
        result = {
            "all": metrics(rows),
            "by_task": {
                task: metrics([r for r in rows if r["task"] == task])
                for task in sorted({r["task"] for r in rows})
            },
            "by_type": {
                kind: metrics([r for r in rows if r["type"] == kind])
                for kind in ["choice", "noul", "score"]
            },
        }
        result["all"]["max_observed_nvml_process_vram_gib"] = run[
            "max_observed_nvml_process_vram_gib"
        ]
        (a.output / "metrics.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        run.update(
            status="completed",
            processed_records=len(rows),
            failed_records=result["all"]["failed_records"],
        )
        out.write_text(json.dumps(run, indent=2) + "\n")
        print(json.dumps(result["all"], indent=2), flush=True)
    except BaseException as error:
        run.update(
            status="failed",
            processed_records=len(rows),
            error=f"{type(error).__name__}: {error}",
        )
        out.write_text(json.dumps(run, indent=2) + "\n")
        raise
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()
