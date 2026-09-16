"""Core reasoning logic for synthetic comparatives in Hindi.

This module follows the required two-layer design: it generates abstract
comparison problems and computes the ground-truth answer independently from any
natural-language realization.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

RELATIONS = (">", "<", "=")
DOMAINS = [
    "height",
    "age",
    "price",
    "weight",
    "quantity",
    "distance",
    "speed",
    "length",
    "size",
    "temperature",
    "duration",
    "score",
]


def compute_relation(left: float, right: float) -> str:
    """Return the symbolic relation between two numeric values."""
    if left > right:
        return ">"
    if left < right:
        return "<"
    return "="


def build_transitive_chain(relations: Sequence[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """Collapse a simple chain of relations into direct implied comparisons.

    Example: [('A','>','B'), ('B','>','C')] -> [('A','>','C')]
    """
    if not relations:
        return []
    chain: List[Tuple[str, str, str]] = []
    for left, op, right in relations:
        if op not in RELATIONS:
            raise ValueError(f"Unsupported relation operator: {op}")
        chain.append((left, op, right))
    if len(chain) == 1:
        return chain
    direct: List[Tuple[str, str, str]] = []
    for i in range(len(chain) - 1):
        a, op1, b = chain[i]
        b2, op2, c = chain[i + 1]
        if b != b2:
            continue
        combined = (a, op1, c)
        if op1 == op2 == ">":
            direct.append((a, ">", c))
        elif op1 == op2 == "<":
            direct.append((a, "<", c))
        elif op1 == ">" and op2 == "=":
            direct.append((a, ">", c))
        elif op1 == "=" and op2 == ">":
            direct.append((a, ">", c))
        elif op1 == "<" and op2 == "=":
            direct.append((a, "<", c))
        elif op1 == "=" and op2 == "<":
            direct.append((a, "<", c))
        else:
            direct.append((a, "=", c))
    return direct or chain


def evaluate_relation_graph(relations: Sequence[Tuple[str, str, str]], start: str, end: str) -> str:
    """Evaluate a basic relation graph between two entities."""
    lookup: Dict[str, Dict[str, str]] = {}
    for left, op, right in relations:
        lookup.setdefault(left, {})[right] = op
        lookup.setdefault(right, {})[left] = reverse_relation(op)

    if start == end:
        return "="
    if end in lookup.get(start, {}):
        return lookup[start][end]

    # simple BFS over linked relations for short chains
    queue: List[str] = [start]
    visited = {start}
    seen: Dict[str, str] = {}
    while queue:
        current = queue.pop(0)
        for nxt, op in lookup.get(current, {}).items():
            if nxt in visited:
                continue
            visited.add(nxt)
            seen[nxt] = op
            if nxt == end:
                return op
            queue.append(nxt)
    return "?"


def reverse_relation(op: str) -> str:
    if op == ">":
        return "<"
    if op == "<":
        return ">"
    return "="


def generate_reasoning_example(domain: str, seed: int | None = None, task: str | None = None) -> Dict[str, Any]:
    """Generate a direct, transitive, or ordering comparison problem."""
    rng = random.Random(seed)
    if domain not in DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}")

    if task is None:
        task = rng.choices(["direct_comparison", "transitive_comparison", "ordering"], weights=[2, 5, 3], k=1)[0]
    if task == "direct_comparison":
        entities = ["A", "B"]
        values = {name: rng.randint(10, 200) for name in entities}
        relations = [("A", compute_relation(values["A"], values["B"]), "B")]
        answer_relation = relations[0][1]
        answer = "A" if answer_relation == ">" else "B" if answer_relation == "<" else "equal"
        question_type = rng.choice(["compare_two", "relation"])
        return {"task": task, "reasoning_type": "direct", "domain": domain, "entities": entities, "values": values, "relations": relations, "question_type": question_type, "query": entities, "answer": answer if question_type == "compare_two" else answer_relation, "answer_relation": answer_relation, "hop_count": 1}

    entity_count = rng.choice([3, 4])
    entities = [chr(ord("A") + index) for index in range(entity_count)]
    start_value = rng.randint(20, 100)
    step = rng.randint(5, 40)
    ascending = rng.choice([True, False])
    values = {entity: start_value + (index * step if ascending else -index * step) for index, entity in enumerate(entities)}
    relations = [(left, compute_relation(values[left], values[right]), right) for left, right in zip(entities, entities[1:])]
    if task == "transitive_comparison":
        left, right = entities[0], entities[-1]
        answer_relation = compute_relation(values[left], values[right])
        answer = left if answer_relation == ">" else right if answer_relation == "<" else "equal"
        return {"task": task, "reasoning_type": "transitive", "domain": domain, "entities": entities, "values": values, "relations": relations, "question_type": "compare_two", "query": [left, right], "answer": answer, "answer_relation": answer_relation, "hop_count": len(relations)}

    question_type = rng.choice(["largest", "smallest", "ordering"])
    ordered = sorted(values, key=values.get, reverse=True)
    answer = ordered if question_type == "ordering" else ordered[0] if question_type == "largest" else ordered[-1]
    return {"task": task, "reasoning_type": "transitive", "domain": domain, "entities": entities, "values": values, "relations": relations, "question_type": question_type, "query": entities, "answer": answer, "answer_relation": ">" if question_type == "largest" else "<" if question_type == "smallest" else "=", "hop_count": len(relations)}
