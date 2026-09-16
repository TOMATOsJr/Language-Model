"""Run generation-based semantic evaluation on the Hindi held-out test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch
from tqdm import tqdm

from hindi.finetuning.config import FinetuneConfig
from hindi.finetuning.generate import generate_answer, load_model
from hindi.finetuning.dataset import build_prompt
from hindi.finetuning.semantic_eval import score_generation, summarize
from hindi.training.config import ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=FinetuneConfig.checkpoint_dir / "best.pt")
    parser.add_argument("--output", type=Path, default=FinetuneConfig.checkpoint_dir / "test_generation_metrics.json")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    cfg = FinetuneConfig()
    model_cfg = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = spm.SentencePieceProcessor(model_file=str(cfg.tokenizer_path))
    pad_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else model_cfg.pad_id
    model = load_model(args.checkpoint, model_cfg, pad_id, device)

    results = []
    with cfg.test_jsonl.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for record in tqdm(records, desc="Evaluating"):
        generated = generate_answer(
            model, tokenizer, build_prompt(record), args.max_new_tokens,
            cfg.max_length, device, temperature=1e-5 if args.greedy else 0.7, top_p=1.0 if args.greedy else 0.9,
        )
        scored = score_generation(record, generated)
        scored.update({"id": record["id"], "question_type": record["question_type"], "gold": record["answer"]})
        results.append(scored)

    report = summarize(results)
    report["checkpoint"] = str(args.checkpoint)
    report["decoding"] = "greedy" if args.greedy else "top_p"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metrics": report, "examples": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
