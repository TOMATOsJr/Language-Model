"""Semantic evaluation for generated Hindi reasoning answers."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

GREATER_WORDS = ("लंबा", "बड़ा", "महंगा", "भारी", "अधिक", "दूर", "तेज", "गर्म")
LESS_WORDS = ("छोटा", "सस्ता", "हल्का", "कम", "पास", "धीमा", "ठंडा")


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def expected_answer(record: Dict[str, Any]) -> Dict[str, Any]:
    values = record["values"]
    entities = record["entities"]
    query = record.get("query", entities[:2] if record["reasoning_type"] == "direct" else [entities[0], entities[-1]])
    question_type = record["question_type"]
    if question_type == "ordering":
        return {"entities": sorted(record["entities"], key=values.get, reverse=True), "relation": "="}
    if question_type == "largest":
        return {"entities": [max(record["entities"], key=values.get)], "relation": ">"}
    if question_type == "smallest":
        return {"entities": [min(record["entities"], key=values.get)], "relation": "<"}
    left, right = query[:2]
    if values[left] > values[right]:
        return {"entities": [left], "relation": ">", "other": right}
    if values[left] < values[right]:
        return {"entities": [right], "relation": ">", "other": left}
    return {"entities": [left, right], "relation": "=", "other": None}


def _relation_from_text(text: str) -> str | None:
    if any(word in text for word in GREATER_WORDS):
        return ">"
    if any(word in text for word in LESS_WORDS):
        return "<"
    if "बराबर" in text or "समान" in text:
        return "="
    return None


def score_generation(record: Dict[str, Any], generated: str) -> Dict[str, Any]:
    text = normalize(generated)
    expected = expected_answer(record)
    expected_entities = expected["entities"]
    entity_hits = [entity for entity in expected_entities if normalize(entity) in text]
    entity_correct = len(entity_hits) == len(expected_entities)
    relation = _relation_from_text(text)
    relation_correct = True if record["question_type"] == "ordering" else relation == expected["relation"]
    if record["question_type"] == "ordering":
        positions = [text.find(normalize(entity)) for entity in expected_entities]
        ordering_correct = all(position >= 0 for position in positions) and positions == sorted(positions)
    else:
        ordering_correct = None
    prompt_echo = any(normalize(part) in text for part in (record.get("context", ""), record.get("question", "")))
    empty = not text
    logical_correct = entity_correct and relation_correct and (ordering_correct is not False)
    return {
        "logical_correct": logical_correct,
        "entity_correct": entity_correct,
        "relation_correct": relation_correct,
        "ordering_correct": ordering_correct,
        "prompt_echo": prompt_echo,
        "empty": empty,
        "generated": generated,
    }


def summarize(results: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    rows = list(results)
    count = max(1, len(rows))
    ordering_rows = [row for row in rows if row["ordering_correct"] is not None]
    ordering_count = max(1, len(ordering_rows))
    return {
        "examples": len(rows),
        "logical_accuracy": sum(row["logical_correct"] for row in rows) / count,
        "entity_accuracy": sum(row["entity_correct"] for row in rows) / count,
        "relation_accuracy": sum(row["relation_correct"] for row in rows) / count,
        "ordering_accuracy": sum(row["ordering_correct"] is True for row in ordering_rows) / ordering_count,
        "prompt_echo_rate": sum(row["prompt_echo"] for row in rows) / count,
        "empty_rate": sum(row["empty"] for row in rows) / count,
    }
