"""Validation helpers for Hindi synthetic generation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def validate_relation_chain(relations: Iterable[Tuple[str, str, str]]) -> bool:
    for _, op, _ in relations:
        if op not in {">", "<", "="}:
            return False
    return True


def validate_example(example: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if example.get("language") != "hindi":
        errors.append("language must be 'hindi'")
    if not example.get("question"):
        errors.append("missing question")
    if not example.get("answer"):
        errors.append("missing answer")
    if not example.get("entities"):
        errors.append("missing entities")
    if example.get("reasoning_type") not in {"direct", "transitive"}:
        errors.append("invalid reasoning type")
    if example.get("question_type") not in {"compare_two", "relation", "largest", "smallest", "ordering"}:
        errors.append("invalid question type")
    if example.get("statement_type") not in {"adjective", "measure", "explicit_relation", "numeric_value"}:
        errors.append("invalid statement type")
    if not validate_relation_chain(example.get("relations", [])):
        errors.append("invalid relation chain")
    if example.get("answer_relation") not in {">", "<", "="}:
        errors.append("invalid answer relation")
    return errors


def validate_dataset(records: Iterable[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    seen_ids = set()
    seen_fingerprints = set()
    for index, record in enumerate(records):
        if record.get("id") in seen_ids:
            errors.append(f"duplicate id: {record.get('id')}")
        seen_ids.add(record.get("id"))

        for err in validate_example(record):
            errors.append(f"record {record.get('id', index)}: {err}")

        fingerprint = (
            record.get("domain"),
            record.get("statement_type"),
            tuple(record.get("entities", [])),
            tuple(tuple(rel) for rel in record.get("relations", [])),
            tuple(sorted(record.get("values", {}).items())),
            record.get("question"),
            record.get("answer_relation"),
            tuple(record.get("answer", [])) if isinstance(record.get("answer"), list) else record.get("answer"),
        )
        if fingerprint in seen_fingerprints:
            errors.append(f"duplicate example: {record.get('id')}")
        seen_fingerprints.add(fingerprint)
    return errors


def validate_split(split: Dict[str, List[Dict[str, Any]]], train: int, val: int, test: int, language: str) -> List[str]:
    errors: List[str] = []
    expected = {"train": train, "val": val, "test": test}
    for key, count in expected.items():
        actual = len(split.get(key, []))
        if actual != count:
            errors.append(f"{key} size mismatch: expected {count}, got {actual}")
    all_ids = []
    for key in ("train", "val", "test"):
        for record in split.get(key, []):
            if record.get("language") != language:
                errors.append(f"wrong language in {key}: {record.get('language')}")
            all_ids.append(record.get("id"))
    if len(set(all_ids)) != len(all_ids):
        errors.append("overlapping IDs across splits")
    return errors
