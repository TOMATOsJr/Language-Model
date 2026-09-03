"""
Autoregressive text generation from a trained Nepali decoder-only checkpoint.
"""

import argparse
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

from nepali.training.config import ModelConfig
from nepali.training.model import TransformerLM

LANGUAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = LANGUAGE_DIR / "tokenizers" / "unigram_8000.model"


def load_model(checkpoint_path, model_cfg, device):
    model = TransformerLM(
        vocab_size=model_cfg.vocab_size,
        dmodel=model_cfg.dmodel,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        dff=model_cfg.dff,
        dropout=0.0,
        pad_id=model_cfg.pad_id,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, context_length, temperature, top_p, device):
    token_ids = [tokenizer.bos_id()] + tokenizer.encode(prompt, out_type=int)

    for _ in range(max_new_tokens):
        # No KV cache: recomputes the full context each step, fine at this model size.
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

        next_token_id = sorted_indices[torch.multinomial(sorted_probs, num_samples=1)].item()
        token_ids.append(next_token_id)

        if next_token_id == tokenizer.eos_id():
            break

    return tokenizer.decode(token_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = ModelConfig()

    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    model = load_model(args.checkpoint, model_cfg, device)

    print(generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    ))


if __name__ == "__main__":
    main()
