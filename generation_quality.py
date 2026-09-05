"""Evaluate Hindi or Nepali generation quality on held-out document prefixes."""

import argparse
import importlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import sacrebleu
import sentencepiece as spm
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm


LANGUAGE_SETTINGS = {
    "hindi": {
        "tokenizer": Path("hindi/tokenizers/unigram_8000.model"),
        "test": Path("hindi/datasets/splits/test.jsonl"),
        "checkpoint": Path("hindi/training/checkpoints/best.pt"),
    },
    "nepali": {
        "tokenizer": Path("nepali/tokenizers/unigram_10000.model"),
        "test": Path("nepali/datasets/splits/test.jsonl"),
        "checkpoint": Path("nepali/training/checkpoints/best.pt"),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("hindi", "nepali"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--test-data", type=Path)
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--prefix-tokens", type=int, default=32)
    parser.add_argument("--continuation-tokens", type=int, default=64)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_model(language, checkpoint_path, device):
    config_module = importlib.import_module(f"{language}.training.config")
    model_module = importlib.import_module(f"{language}.training.model")

    model_config_class = config_module.ModelConfig
    model_class = model_module.TransformerLM

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model_config = model_config_class(
        **checkpoint.get("model_config", asdict(model_config_class()))
    )

    model = model_class(
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


def held_out_examples(
    test_path,
    tokenizer,
    prefix_tokens,
    continuation_tokens,
    max_examples,
):
    """Yield document-level prefix/reference pairs from held-out JSONL text."""
    required_tokens = prefix_tokens + continuation_tokens

    with test_path.open("r", encoding="utf-8") as test_file:
        yielded = 0

        for line in test_file:
            if not line.strip():
                continue

            record = json.loads(line)
            text = record.get("text", "")

            if not isinstance(text, str):
                continue

            text = text.strip()
            token_ids = tokenizer.encode(text, out_type=int)

            if len(token_ids) < required_tokens:
                continue

            prefix_ids = token_ids[:prefix_tokens]
            reference_ids = token_ids[prefix_tokens:required_tokens]

            yield {
                "prefix_ids": prefix_ids,
                "reference_ids": reference_ids,
                "prefix_text": tokenizer.decode(prefix_ids),
                "reference_text": tokenizer.decode(reference_ids),
            }

            yielded += 1
            if yielded >= max_examples:
                return


@torch.no_grad()
def generate_tokens(
    model,
    prefix_ids,
    context_length,
    max_new_tokens,
    eos_id,
    device,
    temperature=None,
):
    """Use greedy decoding when temperature is None; otherwise multinomial sampling."""
    token_ids = list(prefix_ids)
    generated_ids = []

    for _ in range(max_new_tokens):
        context = token_ids[-context_length:]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_ids)[0, -1]

        if temperature is None:
            next_token_id = torch.argmax(logits).item()
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1).item()

        if next_token_id == eos_id:
            break

        token_ids.append(next_token_id)
        generated_ids.append(next_token_id)

    return generated_ids


def ngrams(tokens, n):
    return [
        tuple(tokens[index:index + n])
        for index in range(len(tokens) - n + 1)
    ]


def repetition_rate(generations, n=3):
    """Fraction of generated n-grams that repeat within their own continuation."""
    repeated = 0
    total = 0

    for generated_ids in generations:
        generated_ngrams = ngrams(generated_ids, n)
        counts = Counter(generated_ngrams)

        total += len(generated_ngrams)
        repeated += sum(count - 1 for count in counts.values() if count > 1)

    return repeated / total if total else 0.0


def distinct_n(generations, n):
    all_ngrams = [
        ngram
        for generated_ids in generations
        for ngram in ngrams(generated_ids, n)
    ]

    return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0.0


def evaluate_mode(references, hypotheses, generated_token_sequences):
    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
        tokenize="none",
    )

    chrf = sacrebleu.corpus_chrf(
        hypotheses,
        [references],
        word_order=2,
    )

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=False,
    )
    rouge_l_scores = [
        scorer.score(reference, hypothesis)["rougeL"].fmeasure
        for reference, hypothesis in zip(references, hypotheses)
    ]

    return {
        "bleu": bleu.score,
        "bleu_ngram_order": 4,
        "chrfpp": chrf.score,
        "chrf_character_order": 6,
        "chrf_word_order": 2,
        "rouge_l_f1": sum(rouge_l_scores) / len(rouge_l_scores),
        "repetition_rate_3gram": repetition_rate(generated_token_sequences, n=3),
        "distinct_1": distinct_n(generated_token_sequences, n=1),
        "distinct_2": distinct_n(generated_token_sequences, n=2),
        "generated_tokens": sum(map(len, generated_token_sequences)),
    }


def main():
    args = parse_args()

    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive.")
    if args.prefix_tokens <= 0:
        raise ValueError("--prefix-tokens must be positive.")
    if args.continuation_tokens <= 0:
        raise ValueError("--continuation-tokens must be positive.")
    if any(temperature <= 0 for temperature in args.temperatures):
        raise ValueError("All --temperatures values must be positive.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    settings = LANGUAGE_SETTINGS[args.language]
    checkpoint_path = args.checkpoint or settings["checkpoint"]
    tokenizer_path = args.tokenizer or settings["tokenizer"]
    test_path = args.test_data or settings["test"]

    output_dir = args.output_dir or (
        Path(args.language) / "training" / "generation_quality"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    model, model_config = load_model(args.language, checkpoint_path, device)

    examples = list(
        held_out_examples(
            test_path=test_path,
            tokenizer=tokenizer,
            prefix_tokens=args.prefix_tokens,
            continuation_tokens=args.continuation_tokens,
            max_examples=args.num_examples,
        )
    )

    if not examples:
        raise ValueError(
            "No held-out documents were long enough for the requested prefix "
            "and continuation lengths."
        )

    modes = {"greedy": None}
    modes.update({f"sample_t{temperature:g}": temperature for temperature in args.temperatures})

    all_results = {
        "language": args.language,
        "checkpoint": str(checkpoint_path),
        "tokenizer": str(tokenizer_path),
        "test_data": str(test_path),
        "device": device,
        "num_examples": len(examples),
        "prefix_tokens": args.prefix_tokens,
        "continuation_tokens": args.continuation_tokens,
        "metrics": {},
    }

    qualitative_path = output_dir / "generations.jsonl"
    with qualitative_path.open("w", encoding="utf-8") as output_file:
        for mode_name, temperature in modes.items():
            references = []
            hypotheses = []
            generated_token_sequences = []

            for example in tqdm(examples, desc=f"Generating {mode_name}", unit="example"):
                generated_ids = generate_tokens(
                    model=model,
                    prefix_ids=example["prefix_ids"],
                    context_length=model_config.context_length,
                    max_new_tokens=args.continuation_tokens,
                    eos_id=tokenizer.eos_id(),
                    device=device,
                    temperature=temperature,
                )

                hypothesis = tokenizer.decode(generated_ids)
                references.append(example["reference_text"])
                hypotheses.append(hypothesis)
                generated_token_sequences.append(generated_ids)

                output_file.write(
                    json.dumps(
                        {
                            "mode": mode_name,
                            "temperature": temperature,
                            "prefix": example["prefix_text"],
                            "reference": example["reference_text"],
                            "generation": hypothesis,
                            "generated_token_count": len(generated_ids),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            all_results["metrics"][mode_name] = evaluate_mode(
                references,
                hypotheses,
                generated_token_sequences,
            )

    results_path = output_dir / "generation_metrics.json"
    results_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved aggregate metrics: {results_path}")
    print(f"Saved qualitative outputs: {qualitative_path}")

    for mode_name, metrics in all_results["metrics"].items():
        print(
            f"{mode_name}: "
            f"BLEU-4={metrics['bleu']:.2f}, "
            f"chrF++={metrics['chrfpp']:.2f}, "
            f"ROUGE-L={metrics['rouge_l_f1']:.4f}, "
            f"repeat-3={metrics['repetition_rate_3gram']:.4f}, "
            f"D-1={metrics['distinct_1']:.4f}, "
            f"D-2={metrics['distinct_2']:.4f}"
        )


if __name__ == "__main__":
    main()