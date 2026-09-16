"""Nepali text renderer for comparison tasks."""

from __future__ import annotations

from typing import Any, Dict


def render_question(example: Dict[str, Any]) -> str:
    domain = example["domain"]
    query = example["query"]
    if example.get("question_type") == "largest":
        return f"{', '.join(query)} मध्ये सबैभन्दा ठूलो को हो?"
    if example.get("question_type") == "smallest":
        return f"{', '.join(query)} मध्ये सबैभन्दा सानो को हो?"
    a, b = query[:2]
    if domain == "height":
        return f"{a} र {b} मध्ये को ठुलो छ?"
    if domain == "age":
        return f"{a} र {b} मध्ये को ठूलो छ?"
    if domain == "price":
        return f"{a} र {b} मध्ये को महँगो छ?"
    if domain == "weight":
        return f"{a} र {b} मध्ये को भारी छ?"
    if domain == "quantity":
        return f"{a} र {b} मध्ये को बढी मात्रा छ?"
    return f"{a} र {b} मध्ये को ठूलो छ?"


def render_statement(example: Dict[str, Any]) -> str:
    relations = example["relations"]
    return " ".join(f"{l} {op} {r}" for l, op, r in relations)


def render_sample(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question": render_question(example),
        "statement": render_statement(example),
        "context": render_statement(example),
        "answer": example["answer"],
        "language": "nepali",
        "domain": example["domain"],
    }
