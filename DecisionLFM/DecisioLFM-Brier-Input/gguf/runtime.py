"""Backbone GGUF/CUDA com a cabeça original e os mesmos encoders do PyTorch."""

import ctypes as C
import hashlib
import os
from pathlib import Path
import sys
import threading
import numpy as np
import pynvml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from esquema import prepare_record, format_answer
from modelo import encode

LLAMA_COMMIT = "0ec4027a2a0183de87c007069adddc06e665f7de"
CLEF_MODEL = "Cloudflare/clef-flash"
CLEF_REVISION = "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"


def clef_snapshot():
    from huggingface_hub import snapshot_download

    return snapshot_download(
        CLEF_MODEL, revision=CLEF_REVISION, token=False, local_files_only=True
    )


def gpu_name():
    pynvml.nvmlInit()
    try:
        name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        return name.decode() if isinstance(name, bytes) else name
    finally:
        pynvml.nvmlShutdown()


class Runtime:
    def __init__(
        self,
        family,
        backbone,
        head,
        tokenizer,
        capacity=2048,
        library=None,
        clef_dir=None,
    ):
        self.family = family
        self.capacity = capacity
        if family not in {"decision", "clef"} or capacity < 1:
            raise ValueError("Use family decision/clef e capacity > 0")
        if family == "decision" and (head is None or tokenizer is None):
            raise ValueError("DecisionLFM requer --head e --tokenizer")
        if family == "clef" and head is not None:
            raise ValueError(
                "A cabeça CLEF já está dentro do backbone GGUF; omita --head"
            )
        path = (
            (Path(clef_dir) if clef_dir else clef_snapshot())
            if family == "clef"
            else tokenizer
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if family == "clef":
            sys.path.insert(0, str(path))
            from joint_schema_model import encode_record

            self.encode_clef = encode_record
        self.lib = C.CDLL(
            str(
                Path(library).resolve()
                if library
                else Path(__file__).with_name("libdecisionlfm.so")
            )
        )
        self.lib.df_create.argtypes = [C.c_char_p, C.c_char_p, C.c_int]
        self.lib.df_create.restype = C.c_void_p
        self.lib.df_error.restype = C.c_char_p
        self.lib.df_eval.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_int32),
            C.POINTER(C.c_int32),
            C.c_int,
            C.POINTER(C.c_int32),
            C.c_int,
            C.POINTER(C.c_float),
        ]
        self.lib.df_sync.argtypes = [C.c_void_p]
        self.lib.df_free.argtypes = [C.c_void_p]
        self.lib.df_vram.argtypes = [C.c_void_p]
        self.lib.df_vram.restype = C.c_ulonglong
        self.ptr = self.lib.df_create(
            os.fsencode(backbone), os.fsencode(head) if head else b"", capacity
        )
        if not self.ptr:
            raise RuntimeError(self.lib.df_error().decode())
        if not self.lib.df_vram(self.ptr):
            self.close()
            raise RuntimeError("Nenhum buffer CUDA: compile o backend com GGML_CUDA=ON")

    def close(self):
        if self.ptr:
            self.lib.df_free(self.ptr)
            self.ptr = None

    def sync(self):
        self.lib.df_sync(self.ptr)

    def encoded(self, sample):
        state, questions = prepare_record(sample)
        if self.family == "clef":
            if len(questions) != 1:
                raise ValueError(
                    "This benchmark wrapper expects one question per record"
                )
            enc = self.encode_clef(
                self.tokenizer,
                {"state": sample["state"], "questions": sample["questions"]},
                max_length=sys.maxsize,
            )
            orders = np.zeros(len(enc.input_ids), dtype=np.int32)
            for q in enc.questions:
                orders[slice(*q.question_span)] = q.question_type + 1
                for start, end in q.option_spans:
                    orders[start:end] = 4
            return [
                (
                    q.question_id,
                    questions[q.question_id],
                    enc.input_ids,
                    orders,
                    np.full(len(enc.input_ids), -1, dtype=np.int32),
                    list(q.option_ids),
                )
                for q in enc.questions
            ]
        result = []
        for key, q in questions.items():
            enc = encode(
                self.tokenizer, state, q["instructions"], q["options"], self.capacity
            )
            positions = np.full(len(enc["input_ids"]), -1, dtype=np.int32)
            for i, indices in enumerate(enc["option_positions"]):
                positions[indices] = i
            result.append(
                (
                    key,
                    q,
                    enc["input_ids"],
                    np.zeros(len(positions), dtype=np.int32),
                    positions,
                    enc["option_ids"],
                )
            )
        return result

    def respond(self, sample):
        answers = {}
        tokens = 0
        token_hash = hashlib.sha256()
        for key, q, ids, orders, positions, option_ids in self.encoded(sample):
            ids = np.ascontiguousarray(ids, dtype=np.int32)
            if len(ids) > self.capacity:
                raise ValueError(
                    f"Entrada com {len(ids)} tokens; contexto {self.capacity}, sem truncamento"
                )
            token_hash.update(ids.astype("<i4", copy=False).tobytes())
            token_hash.update(orders.astype("<i4", copy=False).tobytes())
            token_hash.update(positions.astype("<i4", copy=False).tobytes())
            values = np.empty(len(option_ids), dtype=np.float32)
            status = self.lib.df_eval(
                self.ptr,
                ids.ctypes.data_as(C.POINTER(C.c_int32)),
                orders.ctypes.data_as(C.POINTER(C.c_int32)),
                len(ids),
                positions.ctypes.data_as(C.POINTER(C.c_int32)),
                len(values),
                values.ctypes.data_as(C.POINTER(C.c_float)),
            )
            if status:
                raise RuntimeError(self.lib.df_error().decode())
            if not np.all(np.isfinite(values)):
                raise ValueError("Non-finite decision scores")
            ps = np.exp(values.astype(np.float64) - values.max())
            ps /= ps.sum()
            answers[key] = format_answer(q, dict(zip(option_ids, ps.tolist())))
            tokens += len(ids)
        return {
            "answers": answers,
            "usage": {
                "input_tokens": tokens,
                "output_tokens": 0,
                "input_sha256": token_hash.hexdigest(),
            },
        }


class MemorySampler:
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        values = [
            p.usedGpuMemory
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            if p.pid == os.getpid()
        ]
        if values:
            self.peak = max(self.peak, max(values))

    def loop(self):
        while not self.stop.wait(0.002):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()
        pynvml.nvmlShutdown()
