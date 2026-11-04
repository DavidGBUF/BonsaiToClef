"""Exporta DecisionLFM ou CLEF para GGUF F16, Q8_0 e Q4_0."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dados_tipos import digest
from calibracao import load_calibration

LLAMA_COMMIT = "0ec4027a2a0183de87c007069adddc06e665f7de"


def export_decision(checkpoint, destination, quants, gguf):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModel

    training = json.loads((checkpoint / "training.json").read_text())
    if training["status"] != "completed":
        raise ValueError("Use um checkpoint com treinamento concluído")
    config = json.loads((checkpoint / "decision_config.json").read_text())
    temperatures = load_calibration(checkpoint, verify_weights=True)
    if temperatures:
        shutil.copyfile(
            checkpoint / "calibration.json", destination / "calibration.json"
        )
    torch.set_num_threads(8)
    model = AutoModel.from_pretrained(
        config["base_model"],
        revision=config["base_revision"],
        trust_remote_code=True,
        token=False,
        local_files_only=True,
        dtype=torch.float32,
        device_map="cpu",
    )
    if config["has_lora"]:
        model = PeftModel.from_pretrained(
            model, checkpoint / "adapter"
        ).merge_and_unload(safe_merge=True)
    text = model.language_model
    hf = destination / "hf-merged"
    hf.mkdir()
    text_config = text.config.to_dict()
    text_config["architectures"] = ["Lfm3ForCausalLM"]
    (hf / "config.json").write_text(json.dumps(text_config, indent=2) + "\n")
    state = {
        "model." + key: value.detach().contiguous()
        for key, value in text.state_dict().items()
    }
    state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
    save_file(state, hf / "model.safetensors", metadata={"format": "pt"})
    for file in (checkpoint / "tokenizer").iterdir():
        if file.is_file():
            shutil.copyfile(file, hf / file.name)
    head = load_file(checkpoint / "head.safetensors")
    for quant in ["f16", *quants]:
        writer = gguf.GGUFWriter(
            str(destination / f"head-{quant}.gguf"), "decisionlfm_head"
        )
        qtype = getattr(gguf.GGMLQuantizationType, quant.upper())
        for name, tensor in head.items():
            array = tensor.float().numpy().copy()
            if array.ndim == 0:
                array = array.reshape(1)
            if name in {"query.weight", "option.weight"}:
                writer.add_tensor(name, gguf.quantize(array, qtype), raw_dtype=qtype)
            else:
                writer.add_tensor(name, array)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    return hf, {
        "model": config["base_model"],
        "base_revision": config["base_revision"],
        "checkpoint": str(checkpoint.resolve()),
        "head_sha256": digest(checkpoint / "head.safetensors"),
        "lora_merge_dtype": "FP32",
        "training": training,
        "temperatures": temperatures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--clef-dir", type=Path, help="Snapshot CLEF com joint_head.safetensors"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llama-dir", type=Path, default=ROOT / "backends/llama.cpp")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument(
        "--quants", nargs="+", choices=["f16", "q8_0", "q4_0"], default=["q8_0", "q4_0"]
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use um diretório de saída novo")
    llama = args.llama_dir.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(llama), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != LLAMA_COMMIT:
        parser.error(f"Use llama.cpp no commit {LLAMA_COMMIT}")
    build = args.build_dir.resolve() if args.build_dir else llama / "build-decisionlfm"
    converter = llama / "convert_hf_to_gguf.py"
    quantizer = build / "bin/llama-quantize"
    quants = list(dict.fromkeys(quant for quant in args.quants if quant != "f16"))
    if not converter.is_file() or (quants and not quantizer.is_file()):
        parser.error("Prepare o backend com gguf/preparar_backend.sh")
    sys.path.insert(0, str(llama / "gguf-py"))
    import gguf

    args.output_dir.mkdir(parents=True)
    if args.checkpoint:
        hf, info = export_decision(args.checkpoint, args.output_dir, quants, gguf)
    else:
        hf = args.clef_dir.resolve()
        if not (hf / "joint_head.safetensors").is_file():
            raise ValueError("O snapshot CLEF deve incluir joint_head.safetensors")
        info = {
            "model": "CLEF-Flash",
            "source": str(hf),
            "head_sha256": digest(hf / "joint_head.safetensors"),
        }
    env = {
        **os.environ,
        "PYTHONPATH": str(llama / "gguf-py")
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    }
    f16 = args.output_dir / "backbone-f16.gguf"
    subprocess.run(
        [
            sys.executable,
            str(converter),
            str(hf),
            "--outtype",
            "f16",
            "--outfile",
            str(f16),
        ],
        env=env,
        check=True,
    )
    for quant in quants:
        options = (
            ["--pure", "--output-tensor-type", "q4_0", "--token-embedding-type", "q4_0"]
            if quant == "q4_0"
            else []
        )
        subprocess.run(
            [
                str(quantizer),
                *options,
                str(f16),
                str(args.output_dir / f"backbone-{quant}.gguf"),
                quant.upper(),
                "8",
            ],
            check=True,
        )
    files = {}
    for file in sorted(args.output_dir.glob("*.gguf")):
        reader = gguf.GGUFReader(file)
        files[file.name] = {
            "sha256": digest(file),
            "size_bytes": file.stat().st_size,
            "tensor_types": dict(Counter(t.tensor_type.name for t in reader.tensors)),
        }
    info.update(
        status="completed", llama_commit=revision, export_device="cpu", files=files
    )
    (args.output_dir / "export.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Exportação concluída: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
