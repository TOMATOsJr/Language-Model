"""Full Hindi dataset-generation pipeline.

This module creates a deterministic synthetic dataset with train/validation/test
splits and writes JSONL files in a reproducible manner.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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


generator = _load_local_module("hi_dataset_generator", "generator.py")
split_module = _load_local_module("hi_dataset_split", "split.py")
validate_module = _load_local_module("hi_dataset_validate", "validate.py")

generate_batch = generator.generate_batch
split_records = split_module.split_records
validate_dataset = validate_module.validate_dataset
validate_split = validate_module.validate_split


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_dataset(language: str = "hindi", train: int = 20, val: int = 5, test: int = 5, seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    total = train + val + test
    records = generate_batch(count=total, seed=seed)
    split = split_records(records, val_size=val, test_size=test, seed=seed)
    for key in ("train", "val", "test"):
        errors = validate_dataset(split[key])
        if errors:
            raise ValueError(f"Validation failed for {language} {key}: {errors[:5]}")
    split_errors = validate_split(split, train=train, val=val, test=test, language=language)
    if split_errors:
        raise ValueError(f"Split validation failed for {language}: {split_errors[:5]}")
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Hindi synthetic reasoning data.")
    parser.add_argument("--train", type=int, default=20)
    parser.add_argument("--val", type=int, default=5)
    parser.add_argument("--test", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=os.path.join(CURRENT_DIR, "data"))
    args = parser.parse_args()

    split = generate_dataset(language="hindi", train=args.train, val=args.val, test=args.test, seed=args.seed)
    for key, records in split.items():
        out_path = os.path.join(args.output_dir, f"{key}.jsonl")
        write_jsonl(out_path, records)
        print(f"wrote {key}: {len(records)} examples -> {out_path}")


if __name__ == "__main__":
    main()
