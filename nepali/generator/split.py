"""Deterministic train/validation/test splitting for Nepali synthetic data."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Set


def _stable_hash(value: Any, salt: str = "") -> int:
    payload = f"{salt}:{value}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def get_held_out_entity_names(records: Iterable[Dict[str, Any]], seed: int = 42) -> Set[str]:
    names = sorted(
        {entity for record in records for entity in record.get("entities", [])},
        key=lambda item: _stable_hash(f"{seed}:{item}", "entity"),
    )
    if not names:
        return set()
    reserved_count = max(1, min(len(names) // 4, 8))
    return set(names[:reserved_count])


def split_records(
    records: Iterable[Dict[str, Any]],
    val_size: int,
    test_size: int,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split records deterministically while holding out some entity names."""
    record_list = list(records)
    if not record_list:
        return {"train": [], "val": [], "test": []}

    reserved_names = get_held_out_entity_names(record_list, seed=seed)
    eval_candidates = [
        item for item in record_list if set(item.get("entities", [])) & reserved_names
    ]
    remaining = [item for item in record_list if item not in eval_candidates]

    eval_candidates = sorted(
        eval_candidates,
        key=lambda item: _stable_hash(f"{seed}:{item.get('id', 'no-id')}", "eval-sort"),
    )
    val = eval_candidates[: min(val_size, len(eval_candidates))]
    test = eval_candidates[min(val_size, len(eval_candidates)) : min(val_size + test_size, len(eval_candidates))]

    remaining_sorted = sorted(
        remaining,
        key=lambda item: _stable_hash(f"{seed}:{item.get('id', 'no-id')}", "remain-sort"),
    )
    while len(val) < val_size and remaining_sorted:
        candidate = remaining_sorted.pop(0)
        if candidate not in val and candidate not in test:
            val.append(candidate)
    while len(test) < test_size and remaining_sorted:
        candidate = remaining_sorted.pop(0)
        if candidate not in val and candidate not in test:
            test.append(candidate)

    train = [item for item in record_list if item not in val and item not in test]
    return {"train": train, "val": val, "test": test}
