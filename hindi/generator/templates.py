"""Hindi surface constructions for comparison reasoning."""

from __future__ import annotations

from typing import Any, List

LEXICON = {
    "height": ("ऊंचाई", "लंबा", "छोटा"), "age": ("उम्र", "बड़ा", "छोटा"),
    "price": ("कीमत", "महंगा", "सस्ता"), "weight": ("वजन", "भारी", "हल्का"),
    "quantity": ("मात्रा", "अधिक", "कम"), "distance": ("दूरी", "दूर", "पास"),
    "speed": ("गति", "तेज", "धीमा"), "length": ("लंबाई", "लंबा", "छोटा"),
    "size": ("आकार", "बड़ा", "छोटा"), "temperature": ("तापमान", "गर्म", "ठंडा"),
    "duration": ("अवधि", "लंबी", "छोटी"), "score": ("अंक", "अधिक", "कम"),
}


def _parts(domain: str) -> tuple[str, str, str]:
    return LEXICON.get(domain, LEXICON["height"])


def render_statement(domain: str, left: str, right: str, relation: str, statement_type: str = "measure", values: dict[str, int] | None = None) -> str:
    measure, greater, less = _parts(domain)
    if statement_type == "numeric_value" and values is not None:
        unit = {"height": "सेमी", "age": "वर्ष", "price": "रुपये", "weight": "किलो", "quantity": "इकाइयाँ", "distance": "किमी", "speed": "किमी प्रति घंटा", "length": "सेमी", "size": "इकाइयाँ", "temperature": "डिग्री", "duration": "मिनट", "score": "अंक"}.get(domain, "इकाइयाँ")
        return f"{left} की {measure} {values[left]} {unit} है। {right} की {measure} {values[right]} {unit} है।"
    if relation == "=":
        return f"{left} और {right} की {measure} समान है।"
    if relation == "<":
        left, right = right, left
        adjective = greater
    else:
        adjective = greater
    if statement_type == "adjective":
        return f"{left}, {right} से {adjective} है।"
    if statement_type == "explicit_relation":
        return f"{right} की तुलना में {left} की {measure} अधिक है।"
    return f"{left} की {measure} {right} से अधिक है।"


def render_context(domain: str, relations: List[tuple[str, str, str]], statement_type: str = "measure", values: dict[str, int] | None = None) -> str:
    return " ".join(render_statement(domain, left, right, relation, statement_type, values) for left, relation, right in relations)


def render_question(domain: str, query: List[str], question_type: str, relation: str = "") -> str:
    measure, greater, less = _parts(domain)
    entities = "、".join(query)
    if question_type == "relation":
        return f"{query[0]} और {query[1]} के बीच क्या संबंध है?"
    if question_type == "ordering":
        return f"{entities} को {measure} के अनुसार अधिक से कम क्रम में लगाएं।"
    if question_type == "largest":
        return f"{entities} में सबसे {greater} कौन है?"
    if question_type == "smallest":
        return f"{entities} में सबसे {less} कौन है?"
    if relation == "=":
        return f"{query[0]} और {query[1]} बराबर हैं या अलग हैं?"
    return f"{query[0]} और {query[1]} में से कौन {greater} है?"


def render_answer(domain: str, query: List[str], question_type: str, relation: str, answer: Any) -> str:
    measure, greater, less = _parts(domain)
    if question_type == "ordering":
        ordered = "、".join(answer) if isinstance(answer, list) else str(answer)
        return f"{measure} के अनुसार अधिक से कम क्रम: {ordered}।"
    if question_type == "largest":
        return f"{answer} सबसे {greater} है।"
    if question_type == "smallest":
        return f"{answer} सबसे {less} है।"
    if relation == "=":
        return f"{query[0]} और {query[1]} की {measure} समान है।"
    if relation == ">":
        return f"{query[0]}, {query[1]} से {greater} है।"
    return f"{query[1]}, {query[0]} से {greater} है।"
