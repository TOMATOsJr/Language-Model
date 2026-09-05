"""Report cross-entropy, perplexity, and bits per UTF-8 byte for a checkpoint."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nepali.training.config import ModelConfig, TrainConfig
from nepali.training.data import TokenStreamDataset
from nepali.training.model import TransformerLM


LANGUAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = LANGUAGE_DIR / "training" / "checkpoints" / "best.pt"


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint.get("model_config", asdict(ModelConfig())))
    model = TransformerLM(
        vocab_size=model_config.vocab_size,
        dmodel=model_config.dmodel,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        dff=model_config.dff,
        dropout=0.0,
        pad_id=model_config.pad_id,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_config


def count_text_bytes(source_path):
    total_bytes = 0
    with source_path.open("r", encoding="utf-8") as source_file:
        for line in source_file:
            text = json.loads(line.strip()).get("text")
            text = text.strip()
            if text:
                total_bytes += len(text.encode("utf-8"))
    return total_bytes


@torch.no_grad()
def evaluate_split(model, token_path, source_path, tokenizer_path, context_length, batch_size, device, max_tokens, split_name="validation"):
    dataset = TokenStreamDataset(token_path, context_length=context_length)
    evaluated_tokens = min(max_tokens or len(dataset.tokens) - 1, len(dataset.tokens) - 1)
    total_nll = 0.0
    full_blocks, remainder = divmod(evaluated_tokens, context_length)
    for first_block in tqdm(
        range(0, full_blocks, batch_size),
        desc=f"Evaluating {split_name}",
        unit="batch",
        total=math.ceil(full_blocks / batch_size),
    ):
        block_count = min(batch_size, full_blocks - first_block)
        token_blocks = torch.stack([
            torch.tensor(
                dataset.tokens[start:start + context_length + 1], dtype=torch.long, device=device
            )
            for start in range(
                first_block * context_length,
                (first_block + block_count) * context_length,
                context_length,
            )
        ]).to(device, non_blocking=True)
        logits = model(token_blocks[:, :-1])
        total_nll += F.cross_entropy(
            logits.flatten(0, 1), token_blocks[:, 1:].flatten(), reduction="sum"
        ).item()

    if remainder:
        tail_start = full_blocks * context_length
        tail = torch.tensor(
            dataset.tokens[tail_start:tail_start + remainder + 1], dtype=torch.long, device=device
        ).unsqueeze(0)
        logits = model(tail[:, :-1])
        total_nll += F.cross_entropy(logits.flatten(0, 1), tail[:, 1:].flatten(), reduction="sum").item()

    if max_tokens is None:
        total_bytes = count_text_bytes(source_path)
        byte_source = "original_stripped_jsonl_text"
    else:
        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        total_bytes = len(tokenizer.decode(dataset.tokens[:evaluated_tokens + 1].tolist()).encode("utf-8"))
        byte_source = "sentencepiece_decoded_token_prefix"

    loss = total_nll / evaluated_tokens
    return {
        "cross_entropy_loss": loss,
        "perplexity": math.exp(loss),
        "bits_per_byte": total_nll / (math.log(2) * total_bytes),
        "evaluated_tokens": evaluated_tokens,
        "utf8_bytes": total_bytes,
        "bpb_byte_source": byte_source,
    }


def parse_args():
    train_config = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=train_config.micro_batch_size)
    parser.add_argument("--max-tokens", type=int, help="Maximum next-token labels to score per split.")
    parser.add_argument("--output", type=Path, help="Optional JSON file for the metrics.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or (args.max_tokens is not None and args.max_tokens <= 0):
        raise ValueError("--batch-size and --max-tokens must be positive.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_config = load_model(args.checkpoint, device)
    results = {"checkpoint": str(args.checkpoint), "device": device}

    for split_name in ("validation", "test"):
        metadata_path = LANGUAGE_DIR / "datasets" / "tokenized" / f"{split_name}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        results[split_name] = evaluate_split(
            model,
            LANGUAGE_DIR / "datasets" / "tokenized" / f"{split_name}.bin",
            Path(metadata["source"]),
            Path(metadata["tokenizer"]),
            model_config.context_length,
            args.batch_size,
            device,
            args.max_tokens,
            split_name=split_name,
        )
        metrics = results[split_name]
        print(
            f"{split_name}: loss={metrics['cross_entropy_loss']:.6f} "
            f"ppl={metrics['perplexity']:.4f} bpb={metrics['bits_per_byte']:.6f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()