"""Compara peso de score e taxas separadamente com QLoRA; protege MAE e prioriza acurácia."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from calibracao import choose_candidate, weight_hash
from dados_tipos import code_hashes, digest

ROOT = Path(__file__).resolve().parent
CONFIGURATIONS = [
    {
        "name": "controle-peso-2",
        "parent": None,
        "head_lr": 3e-4,
        "lora_lr": 3e-5,
        "score_brier_weight": 2.0,
    },
    {
        "name": "peso-1",
        "parent": "controle-peso-2",
        "head_lr": 3e-4,
        "lora_lr": 3e-5,
        "score_brier_weight": 1.0,
    },
    {
        "name": "peso-0p5",
        "parent": "peso-1",
        "head_lr": 3e-4,
        "lora_lr": 3e-5,
        "score_brier_weight": 0.5,
    },
    {
        "name": "lr-cabeca",
        "parent": "peso-1",
        "head_lr": 1e-3,
        "lora_lr": 3e-5,
        "score_brier_weight": 1.0,
    },
    {
        "name": "lr-lora",
        "parent": "lr-cabeca",
        "head_lr": 1e-3,
        "lora_lr": 1e-4,
        "score_brier_weight": 1.0,
    },
]


def copy_weights(source, destination, calibrated=False):
    ignored = [
        "checkpoints",
        "*.pt",
        "calibration-logits.jsonl",
        "validation-logits.jsonl",
    ]
    if not calibrated:
        ignored.append("calibration.json")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*ignored))
    if weight_hash(source) != weight_hash(destination):
        raise ValueError("Cópia dos pesos divergiu da origem")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--original-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--llama-dir", type=Path)
    p.add_argument("--build-dir", type=Path)
    p.add_argument("--library", type=Path)
    args = p.parse_args()
    if args.output_dir.exists() or args.models_dir.exists():
        p.error("Use diretórios novos")
    if bool(args.llama_dir) != bool(args.build_dir):
        p.error("Informe llama-dir e build-dir juntos")
    args.output_dir.mkdir(parents=True)
    args.models_dir.mkdir(parents=True)
    info = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "models_dir": str(args.models_dir.resolve()),
        "data_manifest_sha256": digest(args.data_dir / "manifest.json"),
        "test_sha256": digest(args.data_dir / "test.jsonl"),
        "configurations": CONFIGURATIONS,
        "epoch_budget_per_configuration": 3,
        "rank": 8,
        "seed": 42,
        "selection_policy": "Validation only: accuracy >= original raw, score MAE <= original raw; highest accuracy, Brier breaks ties. Original raw is the fallback.",
        "calibration_policy": "Fit on calibration half only; score Brier minimized subject to MAE <= T=1. Candidate selection uses the other half.",
        "code_sha256": code_hashes(ROOT),
        "models": [],
        "variants": [],
    }

    def save():
        tmp = args.output_dir / "experiment.tmp"
        tmp.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.output_dir / "experiment.json")

    def run(stage, script, *arguments):
        info["stage"] = stage
        save()
        (args.output_dir / "status.txt").write_text(stage + "\n")
        print(f"stage={stage}", flush=True)
        if stage not in {"export", "summary"} and shutil.which("nvidia-smi"):
            for attempt in range(61):
                processes = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).strip()
                if not processes:
                    break
                info["gpu_wait"] = {
                    "stage": stage,
                    "processes": processes,
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                }
                save()
                (args.output_dir / "status.txt").write_text(f"waiting_gpu:{stage}\n")
                if attempt == 60:
                    raise RuntimeError(
                        f"GPU ocupada por outro processo por mais de 30 minutos: {processes}"
                    )
                time.sleep(30)
            info.pop("gpu_wait", None)
            save()
            (args.output_dir / "status.txt").write_text(stage + "\n")
        with (args.output_dir / f"{stage}.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-u", str(ROOT / script), *map(str, arguments)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def finish_model(name, checkpoint, source=None):
        run(
            f"replay-{name}",
            "verificar_checkpoint.py",
            "--checkpoint",
            checkpoint,
            "--data-dir",
            args.data_dir,
            "--output",
            args.output_dir / f"replay-{name}.json",
        )
        run(
            f"calibrate-{name}",
            "calibrar.py",
            "--checkpoint",
            checkpoint,
            "--data-dir",
            args.data_dir,
            "--protect-score-mae",
        )
        calibration = json.loads((checkpoint / "calibration.json").read_text())
        training = json.loads((checkpoint / "training.json").read_text())
        info["models"].append(
            {
                "name": name,
                "checkpoint": str(checkpoint.resolve()),
                "source_checkpoint": str(source.resolve()) if source else None,
                "selected_epoch": training.get(
                    "selected_epoch", training["history"][-1]["epoch"]
                ),
                "args": training["args"],
            }
        )
        for mode, key in [
            ("raw", "selection_before"),
            ("protected", "selection_after"),
        ]:
            info["variants"].append(
                {
                    "name": f"{name}-{mode}",
                    "model": name,
                    "checkpoint": str(checkpoint.resolve()),
                    "calibrated": mode == "protected",
                    "validation": calibration[key],
                }
            )
        save()

    save()
    try:
        original = args.models_dir / "original"
        copy_weights(args.original_checkpoint, original)
        finish_model("original", original, args.original_checkpoint)
        for config in CONFIGURATIONS:
            name = config["name"]
            checkpoint = args.models_dir / name
            run(
                f"train-{name}",
                "treinar_tipos.py",
                "--data-dir",
                args.data_dir,
                "--output-dir",
                checkpoint,
                "--epochs",
                3,
                "--lora-rank",
                8,
                "--head-lr",
                config["head_lr"],
                "--lora-lr",
                config["lora_lr"],
                "--brier-weight",
                1,
                "--score-brier-weight",
                config["score_brier_weight"],
                "--selection-metric",
                "accuracy",
                "--lr-scheduler",
                "constant",
                "--early-stopping-patience",
                2,
                "--checkpoint-every",
                0,
            )
            finish_model(name, checkpoint)
        reference = next(
            c["validation"] for c in info["variants"] if c["name"] == "original-raw"
        )
        winner = choose_candidate(info["variants"], reference)
        info["selected"] = winner
        # Todos os hiperparâmetros, épocas e temperaturas estão fixados antes do teste.
        (args.output_dir / "selection.json").write_text(
            json.dumps(winner, indent=2) + "\n"
        )
        final = args.models_dir / "selecionado"
        copy_weights(Path(winner["checkpoint"]), final, calibrated=winner["calibrated"])
        info["final_checkpoint"] = str(final.resolve())
        for variant in info["variants"]:
            run(
                f"test-{variant['name']}",
                "avaliar_tipos.py",
                "--data-dir",
                args.data_dir,
                "--checkpoint",
                variant["checkpoint"],
                "--output-dir",
                args.output_dir / variant["name"],
                *([] if variant["calibrated"] else ["--uncalibrated"]),
            )
        if args.llama_dir:
            export = args.models_dir / "selecionado-gguf"
            run(
                "export",
                "gguf/exportar.py",
                "--checkpoint",
                final,
                "--output-dir",
                export,
                "--llama-dir",
                args.llama_dir,
                "--build-dir",
                args.build_dir,
                "--quants",
                "q8_0",
                "q4_0",
            )
            info["export"] = str(export.resolve())
            for quant in ["f16", "q8_0", "q4_0"]:
                options = ["--library", args.library] if args.library else []
                for mode in ["raw", "protected"]:
                    name = f"gguf-{quant}-{mode}"
                    run(
                        name,
                        "gguf/avaliar.py",
                        "--family",
                        "decision",
                        "--backbone",
                        export / f"backbone-{quant}.gguf",
                        "--head",
                        export / f"head-{quant}.gguf",
                        "--tokenizer",
                        final / "tokenizer",
                        "--data-dir",
                        args.data_dir,
                        "--quantization",
                        quant.upper(),
                        "--output",
                        args.output_dir / name,
                        *options,
                        *(["--uncalibrated"] if mode == "raw" else []),
                    )
        run("summary", "resumir_ajustes.py", "--run", args.output_dir)
        info.update(
            status="completed",
            stage="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        (args.output_dir / "status.txt").write_text("completed\n")
    except Exception as error:
        info.update(status="failed", error=f"{type(error).__name__}: {error}")
        (args.output_dir / "status.txt").write_text(f"failed: {info['stage']}\n")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

