"""Prepara JSONL multitarefa com grupos separados e revisões fixas dos datasets."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from esquema import prepare_record


SOURCES = {
    "banking77": ("PolyAI/banking77", None, "choice", "test", False),
    "ag_news": ("fancyzhx/ag_news", None, "choice", "test", False),
    "boolq": ("google/boolq", None, "noul", "validation", False),
    "snli": ("stanfordnlp/snli", None, "noul", "test", False),
    "sst5": ("SetFit/sst5", None, "score", "test", False),
    "yelp": ("Yelp/yelp_review_full", None, "score", "test", False),
    "emotion": ("dair-ai/emotion", "split", "choice", "test", True),
    "rte": ("aps/super_glue", "rte", "noul", "validation", True),
    "amazon": ("SetFit/amazon_reviews_multi_en", None, "score", "test", True),
}
PROMPTS = {
    "banking77": [
        "Identify the customer's banking intent. Choose exactly one option.",
        "Which banking issue does the customer describe?",
        "Select the matching banking intent.",
        "Classify this message using the supplied banking intent descriptions.",
    ],
    "ag_news": [
        "What is the topic of this news report?",
        "Choose the news category.",
        "Which subject best describes this article?",
        "Assign the report to its matching topic.",
    ],
    "boolq": [
        "Based on the passage, answer this question: ",
        "Does the passage support answering yes to: ",
        "Read the passage and decide whether the answer is yes: ",
        "Use the supplied evidence to answer true for yes or false for no: ",
    ],
    "snli": [
        "Does the premise logically imply the hypothesis?",
        "Is the hypothesis entailed by the premise?",
        "Must the hypothesis be true if the premise is true?",
        "Is the hypothesis necessarily supported by the stated premise?",
    ],
    "sst5": [
        "What sentiment does this movie review express?",
        "Rate the review's sentiment.",
        "Select the sentiment level of the text.",
        "Place the movie review on the supplied sentiment scale.",
    ],
    "yelp": [
        "How many stars does this review imply?",
        "Rate the customer's satisfaction from the review.",
        "Choose the matching review rating.",
        "Estimate the star rating expressed by this customer review.",
    ],
    "emotion": ["Which emotion is expressed in this text?"],
    "rte": ["Is the hypothesis necessarily supported by the stated premise?"],
    "amazon": ["Estimate the star rating expressed by this product review."],
}
SENTIMENT = [
    "Very negative sentiment",
    "Negative sentiment",
    "Neutral sentiment",
    "Positive sentiment",
    "Very positive sentiment",
]
STARS = [
    "One star: very dissatisfied",
    "Two stars: dissatisfied",
    "Three stars: neutral or mixed",
    "Four stars: satisfied",
    "Five stars: very satisfied",
]


def read_jsonl(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def code_hashes(directory):
    """Registra nosso código, sem percorrer ambientes, pesos ou backends externos."""
    root = Path(directory)
    return {
        str(path.relative_to(root)): digest(path)
        for folder in (root, root / "gguf", root / "tests")
        for path in sorted(folder.glob("*.py"))
    }


def group_key(task, row):
    if task == "boolq":
        text = row["passage"]
    elif task in {"snli", "rte"}:
        text = row["premise"]
    else:
        text = row["text"]
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def make_record(task, row, names, index, split, rng):
    kind = SOURCES[task][2]
    label = int(row["answer"]) if task == "boolq" else int(row["label"])
    if label < 0:
        return None
    templates = PROMPTS[task]
    instruction = (
        templates[-1]
        if split == "test"
        else templates[rng.randrange(min(3, len(templates)))]
    )
    state = row.get("text")
    if task == "boolq":
        state = {"passage": row["passage"]}
        instruction += row["question"].rstrip(" ?") + "?"
    elif task in {"snli", "rte"}:
        state = {"premise": row["premise"], "hypothesis": row["hypothesis"]}
    elif rng.random() < 0.5:
        state = {"text": state}
    question = {"type": kind, "instructions": instruction}
    if kind == "choice":
        descriptions = names
        if task == "banking77":
            descriptions = [name.replace("_", " ").replace("?", "") for name in names]
        question["criteria"] = dict(zip(names, descriptions))
        target = names[label]
    elif kind == "noul":
        positive = bool(label) if task == "boolq" else label == 0
        target = "true" if positive else "false"
    else:
        question["criteria"] = SENTIMENT[:] if task == "sst5" else STARS[:]
        if split == "train" and rng.random() < 0.5:
            question["criteria"] = [
                "Negative sentiment",
                "Neutral or mixed sentiment",
                "Positive sentiment",
            ]
            label = 0 if label < 2 else 1 if label == 2 else 2
        target = str(label)
    question_id = f"{task}_{index % 19}" if split == "train" else task
    return {
        "id": f"{task}:{split}:{index}",
        "task": task,
        "type": kind,
        "held_out_task": SOURCES[task][4],
        "group": group_key(task, row),
        "state": state,
        "questions": {question_id: question},
        "targets": {question_id: target},
    }


def encoded_record(tokenizer, sample, max_length=8192, rng=None):
    from modelo import encode

    state, questions = prepare_record(sample)
    if set(sample["targets"]) != set(questions):
        raise ValueError("targets deve conter exatamente os IDs das perguntas")
    if len(questions) != 1:
        raise ValueError("Cada exemplo de treino/avaliação deve conter uma pergunta")
    question_id, question = next(iter(questions.items()))
    keys = list(question["options"])
    if rng is not None and question["type"] == "choice":
        rng.shuffle(keys)
    options = {key: question["options"][key] for key in keys}
    target = str(sample["targets"][question_id])
    if target not in options:
        raise ValueError("Alvo ausente das alternativas")
    record = encode(tokenizer, state, question["instructions"], options, max_length)
    record.update(target=keys.index(target), type=sample["type"], task=sample["task"])
    record["id"] = sample.get("id")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-task", type=int, default=2000)
    parser.add_argument("--validation-per-task", type=int, default=300)
    parser.add_argument("--test-per-task", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sources",
        type=Path,
        help="JSON com revisões dos datasets; omita para resolver as revisões atuais",
    )
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or min(
            args.train_per_task,
            args.validation_per_task,
            args.test_per_task,
            args.max_length,
        )
        < 1
    ):
        parser.error("Use diretório novo e limites positivos")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from modelo import MODEL, REVISION

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=True, token=False
    )
    args.output_dir.mkdir(parents=True)
    splits = {key: [] for key in ["train", "validation", "test"]}
    seen = set()
    manifest = {
        "seed": args.seed,
        "max_input_tokens": args.max_length,
        "sources": {},
        "protocol": "Exact normalized document groups disjoint across all splits and sources. "
        "Train and validation from official train by fixed group-hash partition (10% validation). "
        "Test uses official held-out labeled split; last instruction template is test-only. "
        "Overlength inputs are excluded, never truncated. Held-out tasks never enter training.",
    }
    pinned = json.loads(args.sources.read_text()) if args.sources else None
    if pinned is not None and set(pinned) != set(SOURCES):
        raise ValueError("sources deve conter as nove fontes do benchmark")
    api = HfApi(token=False) if pinned is None else None
    for task_index, (task, (repo, config, kind, test_split, held_out)) in enumerate(
        SOURCES.items()
    ):
        if pinned is not None:
            source = pinned[task]
            if source["repository"] != repo or source["config"] != config:
                raise ValueError(f"Fonte diferente do benchmark: {task}")
            revision = source["revision"]
        else:
            revision = api.dataset_info(repo).sha
        names = None
        info = {
            "repository": repo,
            "revision": revision,
            "config": config,
            "type": kind,
            "source_test_split": test_split,
            "held_out_task": held_out,
            "counts": {},
            "excluded": Counter(),
        }
        for source_split in [test_split] if held_out else ["train", test_split]:
            raw = load_dataset(
                repo, config, split=source_split, revision=revision, token=False
            )
            if kind == "choice":
                names = raw.features["label"].names
            labels = (
                names
                if kind == "choice"
                else ["false", "true"]
                if kind == "noul"
                else list(map(str, range(5)))
            )
            destinations = (
                ["test"] if source_split == test_split else ["train", "validation"]
            )
            counts = {key: Counter() for key in destinations}
            limits = {
                "train": args.train_per_task,
                "validation": args.validation_per_task,
                "test": min(
                    args.test_per_task,
                    300 if held_out else args.test_per_task,
                    len(raw),
                ),
            }
            quotas = {
                key: {
                    label: limits[key] // len(labels)
                    + int(i < limits[key] % len(labels))
                    for i, label in enumerate(labels)
                }
                for key in destinations
            }
            if task == "rte":
                quotas["test"] = {label: len(raw) for label in labels}
            rng = random.Random(args.seed + task_index * 100)
            indices = list(range(len(raw)))
            rng.shuffle(indices)
            for index in indices:
                row = raw[index]
                group = group_key(task, row)
                destination = (
                    "test"
                    if source_split == test_split
                    else ("validation" if int(group[:8], 16) % 10 == 0 else "train")
                )
                if group in seen:
                    info["excluded"]["duplicate_group"] += 1
                    continue
                record = make_record(task, row, names, index, destination, rng)
                if record is None:
                    info["excluded"]["unlabeled"] += 1
                    continue
                # Sampling balances the original classes, before optional 5-to-3 ordinal conversion.
                original = (
                    names[row["label"]]
                    if kind == "choice"
                    else (
                        next(iter(record["targets"].values()))
                        if kind == "noul"
                        else str(row["label"])
                    )
                )
                if counts[destination][original] >= quotas[destination][original]:
                    continue
                try:
                    encoded_record(tokenizer, record, args.max_length)
                except ValueError as error:
                    if "limite" not in str(error):
                        raise
                    info["excluded"]["overlength"] += 1
                    continue
                splits[destination].append(record)
                counts[destination][original] += 1
                seen.add(group)
                if all(
                    sum(counts[key].values()) >= limits[key] for key in destinations
                ):
                    break
            for destination in destinations:
                info["counts"][destination] = dict(counts[destination])
                actual = sum(counts[destination].values())
                if actual < limits[destination] and task != "rte":
                    raise ValueError(
                        f"{task}/{destination}: apenas {actual} de {limits[destination]} registros"
                    )
        manifest["sources"][task] = info
        print(task, info["counts"], flush=True)
    for split, records in splits.items():
        random.Random(args.seed).shuffle(records)
        path = args.output_dir / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
            encoding="utf-8",
        )
        manifest[split] = {
            "records": len(records),
            "sha256": digest(path),
            "types": dict(Counter(row["type"] for row in records)),
        }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

