"""Hindi text renderer for comparison tasks."""

from __future__ import annotations

from typing import Any, Dict, List


def render_question(example: Dict[str, Any]) -> str:
    domain = example["domain"]
    query = example["query"]
    if example.get("question_type") == "largest":
        return f"{'、'.join(query)} में सबसे बड़ा कौन है?"
    if example.get("question_type") == "smallest":
        return f"{'、'.join(query)} में सबसे छोटा कौन है?"
    a, b = query[:2]
    if domain == "height":
        return f"{a} और {b} में से कौन लंबा है?"
    if domain == "age":
        return f"{a} और {b} में से कौन बड़ा है?"
    if domain == "price":
        return f"{a} और {b} में से कौन महंगा है?"
    if domain == "weight":
        return f"{a} और {b} में से कौन भारी है?"
    if domain == "quantity":
        return f"{a} और {b} में से कौन अधिक मात्रा रखता है?"
    return f"{a} और {b} में से कौन बड़ा है?"


def render_statement(example: Dict[str, Any]) -> str:
    domain = example["domain"]
    relations = example["relations"]
    if domain == "height":
        return " | ".join(f"{l} {op} {r}" for l, op, r in relations)
    return " ".join(f"{l} {op} {r}" for l, op, r in relations)


def render_sample(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question": render_question(example),
        "statement": render_statement(example),
        "context": render_statement(example),
        "answer": example["answer"],
        "language": "hindi",
        "domain": example["domain"],
    }
