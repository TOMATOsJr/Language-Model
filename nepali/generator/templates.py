"""Nepali surface constructions for comparison reasoning."""

from __future__ import annotations

from typing import Any, List

LEXICON = {
    "height": ("उचाइ", "अग्लो", "होचो"), "age": ("उमेर", "ठूलो", "सानो"),
    "price": ("मूल्य", "महँगो", "सस्तो"), "weight": ("तौल", "भारी", "हल्का"),
    "quantity": ("मात्रा", "धेरै", "थोरै"), "distance": ("दूरी", "टाढा", "नजिक"),
    "speed": ("गति", "छिटो", "ढिलो"), "length": ("लम्बाइ", "लामो", "छोटो"),
    "size": ("आकार", "ठूलो", "सानो"), "temperature": ("तापक्रम", "तातो", "चिसो"),
    "duration": ("अवधि", "लामो", "छोटो"), "score": ("अंक", "धेरै", "कम"),
}


def _parts(domain: str) -> tuple[str, str, str]:
    return LEXICON.get(domain, LEXICON["height"])


def render_statement(domain: str, left: str, right: str, relation: str, statement_type: str = "measure", values: dict[str, int] | None = None) -> str:
    measure, greater, less = _parts(domain)
    if statement_type == "numeric_value" and values is not None:
        unit = {"height": "सेमी", "age": "वर्ष", "price": "रुपैयाँ", "weight": "किलो", "quantity": "एकाइ", "distance": "किमी", "speed": "किमी प्रति घण्टा", "length": "सेमी", "size": "एकाइ", "temperature": "डिग्री", "duration": "मिनेट", "score": "अंक"}.get(domain, "एकाइ")
        return f"{left} को {measure} {values[left]} {unit} छ। {right} को {measure} {values[right]} {unit} छ।"
    if relation == "=":
        return f"{left} र {right} को {measure} बराबर छ।"
    if relation == "<":
        left, right = right, left
        adjective = greater
    else:
        adjective = greater
    if statement_type == "adjective":
        return f"{left}, {right} भन्दा {adjective} छ।"
    if statement_type == "explicit_relation":
        return f"{right} को तुलनामा {left} को {measure} बढी छ।"
    return f"{left} को {measure} {right} भन्दा बढी छ।"


def render_context(domain: str, relations: List[tuple[str, str, str]], statement_type: str = "measure", values: dict[str, int] | None = None) -> str:
    return " ".join(render_statement(domain, left, right, relation, statement_type, values) for left, relation, right in relations)


def render_question(domain: str, query: List[str], question_type: str, relation: str = "") -> str:
    measure, greater, less = _parts(domain)
    entities = ", ".join(query)
    if question_type == "relation":
        return f"{query[0]} र {query[1]} बीचको सम्बन्ध के हो?"
    if question_type == "ordering":
        return f"{entities} लाई {measure} अनुसार धेरैदेखि कम क्रममा मिलाउनुहोस्।"
    if question_type == "largest":
        return f"{entities} मध्ये सबैभन्दा {greater} को हो?"
    if question_type == "smallest":
        return f"{entities} मध्ये सबैभन्दा {less} को हो?"
    if relation == "=":
        return f"{query[0]} र {query[1]} बराबर छन् वा फरक छन्?"
    return f"{query[0]} र {query[1]} मध्ये को {greater} छ?"


def render_answer(domain: str, query: List[str], question_type: str, relation: str, answer: Any) -> str:
    measure, greater, less = _parts(domain)
    if question_type == "ordering":
        ordered = ", ".join(answer) if isinstance(answer, list) else str(answer)
        return f"{measure} अनुसार धेरैदेखि कम क्रम: {ordered}।"
    if question_type == "largest":
        return f"{answer} सबैभन्दा {greater} हो।"
    if question_type == "smallest":
        return f"{answer} सबैभन्दा {less} हो।"
    if relation == "=":
        return f"{query[0]} र {query[1]} को {measure} बराबर छ।"
    if relation == ">":
        return f"{query[0]}, {query[1]} भन्दा {greater} छ।"
    return f"{query[1]}, {query[0]} भन्दा {greater} छ।"
