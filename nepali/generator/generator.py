"""Nepali synthetic comparison generator."""

from __future__ import annotations

import importlib.util
import os
import random
import sys
from typing import Any, Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def _load_local_module(module_name: str, relative_name: str):
    module_path = os.path.join(CURRENT_DIR, relative_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {relative_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


lexicon = _load_local_module("np_lexicon", "lexicon.py")
reasoning = _load_local_module("np_reasoning", "reasoning.py")
templates = _load_local_module("np_templates", "templates.py")

ENTITY_POOLS = lexicon.ENTITY_POOLS
DOMAIN_LEXICON = lexicon.DOMAIN_LEXICON
generate_reasoning_example = reasoning.generate_reasoning_example
render_question = templates.render_question
render_statement = templates.render_statement
render_context = templates.render_context
render_answer = templates.render_answer


def _entity_names_for_index(index: int, category: str, size: int = 4) -> List[str]:
    names = ENTITY_POOLS[category][:]
    rng = random.Random(index + 23)
    return rng.sample(names, k=min(size, len(names)))


def build_record(example: Dict[str, Any], seed: int | None = None, entity_names: List[str] | None = None) -> Dict[str, Any]:
    generic_entities = example["entities"]
    names = entity_names or _entity_names_for_index(seed or 0, example["entity_category"], len(generic_entities))
    mapping = {generic: names[idx] for idx, generic in enumerate(generic_entities)}

    mapped_query = [mapping.get(entity, entity) for entity in example["query"]]
    relation = example["answer_relation"]

    mapped_entities = [mapping.get(entity, entity) for entity in generic_entities]
    mapped_relations = [
        (mapping.get(left_name, left_name), op, mapping.get(right_name, right_name))
        for left_name, op, right_name in example["relations"]
    ]
    question = render_question(example["domain"], mapped_query, example["question_type"], relation)
    mapped_values = {mapping.get(name, name): value for name, value in example["values"].items()}
    statement = render_context(example["domain"], mapped_relations, example["statement_type"], mapped_values)
    raw_answer = example["answer"]
    if isinstance(raw_answer, list):
        mapped_answer = [mapping.get(item, item) for item in raw_answer]
    elif raw_answer in {">", "<", "="}:
        mapped_answer = raw_answer
    else:
        mapped_answer = "बराबर" if raw_answer == "equal" else mapping.get(raw_answer, raw_answer)
    answer = render_answer(example["domain"], mapped_query, example["question_type"], relation, mapped_answer)

    return {
        "id": f"np_{seed or 0:05d}",
        "language": "nepali",
        "task": example["task"],
        "reasoning_type": example["reasoning_type"],
        "domain": example["domain"],
        "entity_category": example["entity_category"],
        "hop_count": example["hop_count"],
        "entities": mapped_entities,
        "relations": mapped_relations,
        "question_type": example["question_type"],
        "statement_type": example["statement_type"],
        "values": mapped_values,
        "question": question,
        "statement": statement,
        "context": statement,
        "answer": answer,
        "answer_relation": example["answer_relation"],
        "template_id": f"np_{example['domain']}_{example['task']}",
    }


def generate_batch(count: int = 20, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    results: List[Dict[str, Any]] = []
    domains = list(DOMAIN_LEXICON)
    seen_fingerprints = set()
    index = 0
    while len(results) < count:
        domain = domains[index % len(domains)]
        task = ("direct_comparison", "transitive_comparison", "ordering")[index % 3]
        example = generate_reasoning_example(domain, seed=rng.randint(0, 10_000) + index, task=task)
        category = random.Random(index + 101).choice(DOMAIN_LEXICON[domain]["categories"])
        example["entity_category"] = category
        if task == "direct_comparison" and index % 2 == 0:
            example["statement_type"] = "numeric_value"
            example["question_type"] = "compare_two"
        else:
            example["statement_type"] = ("adjective", "measure", "explicit_relation")[index % 3]
        entity_names = _entity_names_for_index(index, category, len(example["entities"]))
        result = build_record(example, seed=index, entity_names=entity_names)
        fingerprint = (
            tuple(result["entities"]),
            tuple(tuple(rel) for rel in result["relations"]),
            result["domain"],
            result["answer_relation"],
            result["question"],
            result["statement"],
        )
        if fingerprint in seen_fingerprints:
            index += 1
            continue
        seen_fingerprints.add(fingerprint)
        result["id"] = f"np_{index:05d}"
        results.append(result)
        index += 1
    return results


def main() -> None:
    for item in generate_batch(5, seed=7):
        print(item["question"])
        print(item["statement"])
        print("answer:", item["answer"])
        print("---")


if __name__ == "__main__":
    main()
