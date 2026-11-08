"""Busca curta com QLoRA: até três épocas por configuração, seleção e calibração separadas."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from dados_tipos import code_hashes, digest

ROOT = Path(__file__).resolve().parent
CONFIGURATIONS = [
    {"name": "lr-menor", "head_lr": 3e-4, "lora_lr": 3e-5, "score_brier_weight": 1.0},
    {
        "name": "score-brier-2",
        "head_lr": 3e-4,
        "lora_lr": 3e-5,
        "score_brier_weight": 2.0,
    },
    {
        "name": "score-brier-2-lr-menor",
        "head_lr": 1e-4,
        "lora_lr": 1e-5,
        "score_brier_weight": 2.0,
    },
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--llama-dir", type=Path, help="Opcional: exporta e testa F16/Q8/Q4")
    p.add_argument("--build-dir", type=Path)
    p.add_argument("--library", type=Path)
    p.add_argument(
        "--historical",
        type=Path,
        help="comparison-10ep.json com os controles históricos de três épocas",
    )
    args = p.parse_args()
    if args.output_dir.exists() or args.models_dir.exists():
        p.error("Use diretórios novos para preservar os resultados anteriores")
    if bool(args.llama_dir) != bool(args.build_dir):
        p.error("Informe llama-dir e build-dir juntos")
    args.output_dir.mkdir(parents=True)
    args.models_dir.mkdir(parents=True)
    info = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "epoch_budget_per_configuration": 3,
        "lora_rank": 8,
        "brier_weight": 1.0,
        "configurations": CONFIGURATIONS,
        "candidates": [],
        "data_dir": str(args.data_dir.resolve()),
        "data_manifest_sha256": digest(args.data_dir / "manifest.json"),
        "test_sha256": digest(args.data_dir / "test.jsonl"),
        "historical": str(args.historical.resolve()) if args.historical else None,
        "historical_sha256": digest(args.historical) if args.historical else None,
        "code_sha256": code_hashes(ROOT),
        "protocol": "All candidates use QLoRA 4-bit NF4 and the same selection half of validation; temperature uses the other half. Test runs only after configuration/epoch/temperature are fixed.",
    }

    def save():
        path = args.output_dir / "experiment.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
        tmp.replace(path)

    def run(stage, script, *arguments):
        info["stage"] = stage
        save()
        (args.output_dir / "status.txt").write_text(stage + "\n")
        print(f"stage={stage}", flush=True)
        with (args.output_dir / f"{stage}.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-u", str(ROOT / script), *map(str, arguments)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    save()
    try:
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
                1.0,
                "--score-brier-weight",
                config["score_brier_weight"],
                "--early-stopping-patience",
                2,
                "--checkpoint-every",
                0,
            )
            run(
                f"replay-{name}",
                "verificar_checkpoint.py",
                "--data-dir",
                args.data_dir,
                "--checkpoint",
                checkpoint,
                "--output",
                args.output_dir / f"replay-{name}.json",
            )
            training = json.loads((checkpoint / "training.json").read_text())
            if training["status"] != "completed" or training["completed_epochs"] > 3:
                raise ValueError(f"Treino inválido: {name}")
            info["candidates"].append(
                {
                    "name": name,
                    "checkpoint": str(checkpoint.resolve()),
                    "selected_epoch": training["selected_epoch"],
                    "completed_epochs": training["completed_epochs"],
                    "selection_brier": training["best_validation_brier"],
                    "selection_score_brier": training["history"][
                        training["selected_epoch"] - 1
                    ]["validation"]["selection"]["types"]["score"]["brier"],
                }
            )
            save()
        winner = min(info["candidates"], key=lambda x: x["selection_brier"])
        info["selected"] = winner
        # O vencedor está fixado antes de ajustar T ou ler previsões de teste.
        (args.output_dir / "selection.json").write_text(
            json.dumps(winner, indent=2) + "\n"
        )
        checkpoint = Path(winner["checkpoint"])
        run(
            "calibration",
            "calibrar.py",
            "--data-dir",
            args.data_dir,
            "--checkpoint",
            checkpoint,
        )
        for name, options in [
            ("pytorch-uncalibrated", ["--uncalibrated"]),
            ("pytorch-calibrated", []),
        ]:
            run(
                name,
                "avaliar_tipos.py",
                "--data-dir",
                args.data_dir,
                "--checkpoint",
                checkpoint,
                "--output-dir",
                args.output_dir / name,
                *options,
            )
        if args.llama_dir:
            export = args.models_dir / (winner["name"] + "-gguf")
            run(
                "export",
                "gguf/exportar.py",
                "--checkpoint",
                checkpoint,
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
                for mode in ["uncalibrated", "calibrated"]:
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
                        checkpoint / "tokenizer",
                        "--data-dir",
                        args.data_dir,
                        "--quantization",
                        quant.upper(),
                        "--output",
                        args.output_dir / name,
                        *options,
                        *(["--uncalibrated"] if mode == "uncalibrated" else []),
                    )
        run("summary", "resumir_calibracao.py", "--run", args.output_dir)
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

