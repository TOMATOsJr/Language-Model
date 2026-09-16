"""Instruction-style dataset that masks the loss to only the answer tokens."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def build_prompt(record: dict) -> str:
    return f"{record['context']} {record['question']} उत्तर:"


class ReasoningDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []
        with Path(jsonl_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        prompt = build_prompt(record)
        answer = " " + record["answer"]

        prompt_ids = [self.tokenizer.bos_id()] + self.tokenizer.encode(prompt, out_type=int)
        answer_ids = self.tokenizer.encode(answer, out_type=int) + [self.tokenizer.eos_id()]

        # Keep the tail of the prompt (closest to the answer) if the pair overflows.
        overflow = len(prompt_ids) + len(answer_ids) - self.max_length
        if overflow > 0:
            prompt_ids = prompt_ids[overflow:]

        input_ids = (prompt_ids + answer_ids)[: self.max_length]
        labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[: self.max_length]

        return {"input_ids": input_ids, "labels": labels}


def collate_batch(batch: list[dict], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    for row, item in enumerate(batch):
        length = len(item["input_ids"])
        input_ids[row, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[row, :length] = torch.tensor(item["labels"], dtype=torch.long)
    return input_ids, labels
