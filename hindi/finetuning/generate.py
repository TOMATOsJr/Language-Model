"""Generates an answer for a reasoning prompt using a finetuned Hindi checkpoint."""

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

from hindi.finetuning.config import FinetuneConfig
from hindi.finetuning.dataset import build_prompt
from hindi.training.config import ModelConfig
from hindi.training.model import TransformerLM


def load_model(checkpoint_path, model_cfg, pad_id, device):
    model = TransformerLM(
        vocab_size=model_cfg.vocab_size,
        dmodel=model_cfg.dmodel,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        dff=model_cfg.dff,
        dropout=0.0,
        pad_id=pad_id,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def generate_answer(model, tokenizer, prompt, max_new_tokens, context_length, device, temperature=0.7, top_p=0.9):
    token_ids = [tokenizer.bos_id()] + tokenizer.encode(prompt, out_type=int)
    prompt_len = len(token_ids)

    for _ in range(max_new_tokens):
        context = token_ids[-context_length:]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_ids)[0, -1] / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0.0
        sorted_probs /= sorted_probs.sum()

        next_id = sorted_indices[torch.multinomial(sorted_probs, num_samples=1)].item()
        token_ids.append(next_id)
        if next_id == tokenizer.eos_id():
            break

    return tokenizer.decode(token_ids[prompt_len:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=FinetuneConfig.checkpoint_dir / "best.pt")
    parser.add_argument("--tokenizer", type=Path, default=FinetuneConfig.tokenizer_path)
    parser.add_argument("--context", type=str)
    parser.add_argument("--question", type=str)
    parser.add_argument("--test-index", type=int, help="Index into test.jsonl to sample a record instead of manual input.")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--context-length", type=int, default=224)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    if args.test_index is not None:
        with FinetuneConfig.test_jsonl.open("r", encoding="utf-8") as handle:
            record = json.loads(handle.readlines()[args.test_index])
        prompt = build_prompt(record)
        print("gold answer:", record["answer"])
    else:
        if args.context is None or args.question is None:
            raise SystemExit("Provide --context and --question, or --test-index.")
        prompt = build_prompt({"context": args.context, "question": args.question})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = ModelConfig()
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    pad_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else model_cfg.pad_id
    model = load_model(args.checkpoint, model_cfg, pad_id, device)

    answer = generate_answer(
        model, tokenizer, prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        device=device,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print("prompt:", prompt)
    print("generated answer:", answer)


if __name__ == "__main__":
    main()
