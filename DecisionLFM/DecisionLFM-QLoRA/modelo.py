"""Backbone LFM3 com QLoRA (4-bit NF4 + Double Quantization) + pontuador compartilhado, sem geração."""

import hashlib
import json
import math
import os
from pathlib import Path

# Os checkpoints usados neste experimento são públicos.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from esquema import format_answer, prepare_record
from calibracao import load_calibration, probabilities

MODEL = "LiquidAI/LFM3-VL-2B"
REVISION = "f83d2b0a2296aaa277f394990a9b8b949ddf9691"
LORA_TARGETS = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)"
)
VISION_SKIP_MODULES = ("vision_tower", "multi_modal_projector")
DEFAULT_QUANTIZATION = {
    "enabled": True,
    "method": "bitsandbytes_4bit",
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "llm_int8_skip_modules": list(VISION_SKIP_MODULES),
}


def get_device(quantize_4bit=True):
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.device("cuda")
    if not quantize_4bit and torch.backends.mps.is_available():
        return torch.device("mps")
    if quantize_4bit:
        raise RuntimeError(
            "QLoRA 4-bit (bitsandbytes NF4) requer GPU CUDA com suporte a BF16"
        )
    raise RuntimeError("Este experimento requer CUDA/BF16 ou Apple Metal (MPS)")


def make_bnb_config(
    quant_type="nf4",
    double_quant=True,
    compute_dtype="bfloat16",
    skip_modules=VISION_SKIP_MODULES,
):
    """Cria a configuração BitsAndBytesConfig 4-bit padrão do QLoRA."""
    if quant_type not in {"nf4", "fp4"}:
        raise ValueError("quant_type deve ser nf4 ou fp4")
    if not isinstance(double_quant, bool):
        raise ValueError("double_quant deve ser booleano")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        torch.bfloat16: torch.bfloat16,
        torch.float16: torch.float16,
        torch.float32: torch.float32,
    }
    if compute_dtype not in dtype_map:
        raise ValueError("compute_dtype inválido para QLoRA")
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=dtype_map[compute_dtype],
        llm_int8_skip_modules=list(skip_modules) if skip_modules else None,
    )


def quantization_metadata(
    enabled=True,
    quant_type="nf4",
    double_quant=True,
    compute_dtype="bfloat16",
    skip_modules=VISION_SKIP_MODULES,
):
    if not enabled:
        return {"enabled": False, "method": "none", "dtype": "bfloat16"}
    if quant_type not in {"nf4", "fp4"}:
        raise ValueError("quant_type deve ser nf4 ou fp4")
    if compute_dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("compute_dtype deve ser bfloat16, float16 ou float32")
    return {
        "enabled": True,
        "method": "bitsandbytes_4bit",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": quant_type,
        "bnb_4bit_use_double_quant": bool(double_quant),
        "bnb_4bit_compute_dtype": compute_dtype,
        "llm_int8_skip_modules": list(skip_modules) if skip_modules else [],
    }


def encode(tokenizer, state, question, options, max_length=8192):
    """Uma pergunta, >= 2 alternativas {id: descrição}; sem rótulo na entrada."""
    if (
        not isinstance(state, str)
        or not isinstance(question, str)
        or not question.strip()
    ):
        raise ValueError(
            "state e question devem ser textos; question não pode ser vazia"
        )
    if len(options) < 2 or any(
        not isinstance(k, str) or not isinstance(v, str) or not v.strip()
        for k, v in options.items()
    ):
        raise ValueError("Informe ao menos duas alternativas {id: descrição não vazia}")
    content = question + "\nOptions:\n"
    spans = []
    for description in options.values():
        content += "- "
        start = len(content)
        content += description
        spans.append((start, len(content)))
        content += "\n"
    content += "\nState:\n" + state + "\nChoose exactly one option."
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    offset = prompt.index(content)
    tokens = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    if len(tokens["input_ids"]) > max_length:
        raise ValueError(
            f"Entrada com {len(tokens['input_ids'])} tokens; limite {max_length}."
        )
    positions = []
    for start, end in spans:
        indices = [
            i
            for i, (a, b) in enumerate(tokens["offset_mapping"])
            if b > a and a < offset + end and b > offset + start
        ]
        if not indices:
            raise ValueError("Alternativa sem tokens no prompt")
        positions.append(indices)
    return {
        "input_ids": tokens["input_ids"],
        "option_positions": positions,
        "option_ids": list(options),
    }


def collate(records, pad_id, device="cpu"):
    """Padding à direita; máscara de pooling contém apenas tokens das alternativas."""
    batch, length = len(records), max(len(r["input_ids"]) for r in records)
    choices = max(len(r["option_ids"]) for r in records)
    ids = torch.full((batch, length), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    weights = torch.zeros(batch, choices, length)
    valid = torch.zeros(batch, choices, dtype=torch.bool)
    for i, record in enumerate(records):
        n = len(record["input_ids"])
        ids[i, :n] = torch.tensor(record["input_ids"])
        mask[i, :n] = 1
        for j, positions in enumerate(record["option_positions"]):
            weights[i, j, positions] = 1 / len(positions)
            valid[i, j] = True
    return {
        "input_ids": ids.to(device),
        "attention_mask": mask.to(device),
        "option_weights": weights.to(device),
        "option_mask": valid.to(device),
    }


class DecisionHead(nn.Module):
    def __init__(self, hidden_size, head_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, head_size, bias=False)
        self.option = nn.Linear(hidden_size, head_size, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, query, options, valid):
        q = F.normalize(self.query(self.norm(query.float())), dim=-1)
        k = F.normalize(self.option(self.norm(options.float())), dim=-1)
        scores = torch.einsum("bd,bcd->bc", q, k) * self.log_scale.exp().clamp(max=100)
        return scores.masked_fill(~valid, -torch.inf)


class DecisionLFM(nn.Module):
    def __init__(
        self,
        backbone,
        tokenizer,
        head_size=256,
        max_length=8192,
        quantization=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head_size = head_size
        device = next(backbone.parameters()).device
        self.head = DecisionHead(
            backbone.config.text_config.hidden_size, head_size
        ).to(device=device, dtype=torch.float32)
        self.has_lora = False
        self.quantization = dict(
            quantization if quantization is not None else DEFAULT_QUANTIZATION
        )
        self.temperatures = {}
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    @property
    def is_quantized_4bit(self):
        return bool(self.quantization.get("enabled", False))

    def train(self, mode=True):
        super().train(mode)
        if not self.has_lora:
            self.backbone.eval()
        return self

    def add_lora(self, rank=8, alpha=None, dropout=0.05):
        if rank < 1 or (alpha is not None and alpha < 1) or not (0.0 <= dropout < 1.0):
            raise ValueError("Parâmetros LoRA inválidos")
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        lora_alpha = 2 * rank if alpha is None else alpha
        if self.is_quantized_4bit or getattr(
            self.backbone, "is_loaded_in_4bit", False
        ):
            self.backbone = prepare_model_for_kbit_training(
                self.backbone,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        else:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.backbone = get_peft_model(
            self.backbone,
            LoraConfig(
                r=rank,
                lora_alpha=lora_alpha,
                lora_dropout=dropout,
                bias="none",
                target_modules=LORA_TARGETS,
            ),
        )
        self.has_lora = True

    def forward(self, batch):
        # Sem LM head, logits do vocabulário, KV cache ou generate().
        hidden = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        query_index = batch["attention_mask"].sum(-1) - 1
        query = hidden[torch.arange(hidden.shape[0], device=hidden.device), query_index]
        options = torch.bmm(batch["option_weights"].to(hidden.dtype), hidden)
        return self.head(query, options, batch["option_mask"])

    @torch.inference_mode()
    def decidir(self, state, question, options, temperature=1.0):
        if self.training:
            raise ValueError("Use model.eval() antes de decidir")
        record = encode(self.tokenizer, state, question, options, self.max_length)
        batch = collate(
            [record], self.tokenizer.pad_token_id, next(self.head.parameters()).device
        )
        ps = probabilities(self(batch)[0], temperature)
        if not torch.isfinite(ps).all():
            raise ValueError("Probabilidades não finitas")
        values = dict(zip(record["option_ids"], ps.tolist()))
        prediction = max(values, key=values.get)
        return {
            "prediction": prediction,
            "confidence": values[prediction],
            "probabilities": values,
            "input_tokens": len(record["input_ids"]),
        }

    def responder(self, record):
        """Responde cada pergunta com a interface existente, sem alterar os pesos."""
        state, questions = prepare_record(record)
        answers, input_tokens = {}, 0
        for key, question in questions.items():
            temperature = getattr(self, "temperatures", {}).get(question["type"], 1.0)
            result = self.decidir(
                state,
                question["instructions"],
                question["options"],
                temperature=temperature,
            )
            answers[key] = format_answer(question, result["probabilities"])
            input_tokens += result["input_tokens"]
        return {
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                k: v.detach().cpu().contiguous()
                for k, v in self.head.state_dict().items()
            },
            directory / "head.safetensors",
        )
        if self.has_lora:
            self.backbone.save_pretrained(
                directory / "adapter",
                safe_serialization=True,
                save_embedding_layers=False,
            )
        self.tokenizer.save_pretrained(directory / "tokenizer")
        config = {
            "format_version": 1,
            "base_model": MODEL,
            "base_revision": REVISION,
            "head_size": self.head_size,
            "max_length": self.max_length,
            "has_lora": self.has_lora,
            "quantization": self.quantization,
        }
        (directory / "decision_config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )

    @classmethod
    def load(cls, directory, calibrated=True, quantize_4bit=None):
        from peft import PeftModel
        from transformers import AutoTokenizer

        directory = Path(directory)
        config = json.loads((directory / "decision_config.json").read_text())
        if config["format_version"] != 1:
            raise ValueError("Formato de checkpoint desconhecido")
        saved_quant = config.get("quantization")
        if quantize_4bit is None:
            use_4bit = (
                bool(saved_quant.get("enabled", True))
                if isinstance(saved_quant, dict)
                else False
            )
        else:
            use_4bit = bool(quantize_4bit)
        quant_kwargs = {}
        if use_4bit and isinstance(saved_quant, dict) and saved_quant.get("enabled"):
            quant_kwargs = {
                "quant_type": saved_quant.get("bnb_4bit_quant_type", "nf4"),
                "double_quant": saved_quant.get("bnb_4bit_use_double_quant", True),
                "compute_dtype": saved_quant.get("bnb_4bit_compute_dtype", "bfloat16"),
                "skip_modules": saved_quant.get(
                    "llm_int8_skip_modules", VISION_SKIP_MODULES
                ),
            }
        model = create(
            config["head_size"],
            config["max_length"],
            config["base_model"],
            config["base_revision"],
            quantize_4bit=use_4bit,
            **quant_kwargs,
        )
        model.tokenizer = AutoTokenizer.from_pretrained(
            directory / "tokenizer", local_files_only=True
        )
        if config["has_lora"]:
            model.backbone = PeftModel.from_pretrained(
                model.backbone, directory / "adapter", is_trainable=False
            )
            model.has_lora = True
        head_device = next(model.head.parameters()).device
        model.head.load_state_dict(
            load_file(directory / "head.safetensors", device="cpu")
        )
        model.head.to(device=head_device, dtype=torch.float32)
        model.temperatures = (
            load_calibration(directory, verify_weights=True) if calibrated else {}
        )
        return model.eval()


def create(
    head_size=256,
    max_length=8192,
    model_id=MODEL,
    revision=REVISION,
    quantize_4bit=True,
    quant_type="nf4",
    double_quant=True,
    compute_dtype="bfloat16",
    skip_modules=VISION_SKIP_MODULES,
):
    from transformers import AutoModel, AutoTokenizer

    device = get_device(quantize_4bit=quantize_4bit)
    torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        code_revision=revision,
        token=False,
        trust_remote_code=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("O codificador exige tokenizer rápido com offsets")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_info = quantization_metadata(
        enabled=quantize_4bit,
        quant_type=quant_type,
        double_quant=double_quant,
        compute_dtype=compute_dtype,
        skip_modules=skip_modules,
    )
    if quantize_4bit:
        bnb_config = make_bnb_config(
            quant_type=quant_type,
            double_quant=double_quant,
            compute_dtype=compute_dtype,
            skip_modules=skip_modules,
        )
        backbone = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            code_revision=revision,
            trust_remote_code=True,
            token=False,
            dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map={"": torch.cuda.current_device()},
        )
    else:
        backbone = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            code_revision=revision,
            trust_remote_code=True,
            token=False,
            dtype=torch.bfloat16,
        ).to(device)
    return DecisionLFM(
        backbone,
        tokenizer,
        head_size=head_size,
        max_length=max_length,
        quantization=quant_info,
    )


def checkpoint_hash(directory):
    """Identifica o checkpoint final; ignora os checkpoints intermediários."""
    digest = hashlib.sha256()
    for path in sorted(Path(directory).rglob("*")):
        if (
            path.is_file()
            and "checkpoints" not in path.relative_to(directory).parts
            and path.suffix in {".safetensors", ".json", ".jinja"}
        ):
            digest.update(str(path.relative_to(directory)).encode())
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()

