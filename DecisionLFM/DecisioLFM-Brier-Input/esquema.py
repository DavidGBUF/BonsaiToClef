"""Entrada JSON e respostas choice, noul e score."""

import json


def prepare_record(record):
    if not isinstance(record, dict) or "state" not in record:
        raise ValueError("Informe um objeto JSON com state e questions")
    state = record["state"]
    if not isinstance(state, str):
        try:
            state = json.dumps(
                state,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("state deve ser texto ou um valor JSON válido") from error
    questions = record.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions deve ser um objeto não vazio")
    prepared = {}
    for question_id, question in questions.items():
        if (
            not isinstance(question_id, str)
            or not question_id.strip()
            or not isinstance(question, dict)
        ):
            raise ValueError("Cada pergunta deve ter um ID textual e uma especificação")
        kind = question.get("type")
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = question_id
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(f"{question_id}: instructions deve ser texto")
        criteria = question.get("criteria")
        if kind == "noul":
            options = {
                "true": "The proposition is true or the answer is yes.",
                "false": "The proposition is false or the answer is no.",
            }
            if criteria is not None and (
                not isinstance(criteria, dict) or set(criteria) - set(options)
            ):
                raise ValueError(f"{question_id}: noul aceita critérios true e false")
            options.update(criteria or {})
        elif kind == "score":
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise ValueError(
                    f"{question_id}: score requer uma lista com ao menos dois níveis"
                )
            options = {str(i): description for i, description in enumerate(criteria)}
        elif kind == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(
                    f"{question_id}: choice requer ao menos duas alternativas"
                )
            options = dict(criteria)
        else:
            raise ValueError(f"{question_id}: type deve ser choice, noul ou score")
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in options.items()
        ):
            raise ValueError(
                f"{question_id}: IDs e descrições devem ser textos não vazios"
            )
        if kind == "choice":
            options = dict(sorted(options.items()))
        prepared[question_id] = {
            "type": kind,
            "instructions": instructions,
            "options": options,
        }
    return state, prepared


def format_answer(question, probabilities):
    kind = question["type"]
    if kind == "noul":
        return {"type": kind, "noul": probabilities["true"]}
    answer = {
        "type": kind,
        "confidence": max(probabilities.values()),
        "probabilities": probabilities,
    }
    if kind == "score":
        answer.update(
            score=sum(i * probabilities[str(i)] for i in range(len(probabilities))),
            legend=question["options"],
        )
    else:
        answer["choice"] = max(probabilities, key=probabilities.get)
    return answer
