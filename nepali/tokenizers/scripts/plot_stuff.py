#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import sentencepiece as spm
from tqdm import tqdm


# ============================================================
# CONSTANTS
# ============================================================

# This is NOT assumed to be 0 anymore.
# We get the actual UNK ID from the SentencePiece model.
# ============================================================


# ============================================================
# JSONL TEXT STREAM
# ============================================================

def jsonl_text_iterator(path: Path):
    """
    Stream the `text` field directly from a JSONL file.

    The complete JSONL file is never loaded into memory.
    """

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            text = record.get(
                "text",
                "",
            )

            if not isinstance(
                text,
                str,
            ):
                continue

            text = text.strip()

            if text:
                yield text


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_tokenizer(
    model_path: Path,
    dataset_path: Path,
    top_k: int,
    num_examples: int,
):
    """
    Evaluate one SentencePiece model directly on JSONL.

    Fertility:
        total SentencePiece tokens / whitespace words

    UNK rate:
        UNK tokens / total SentencePiece tokens

    Average characters per token:
        non-whitespace characters / total tokens
    """

    sp = spm.SentencePieceProcessor(
        model_file=str(model_path)
    )

    unk_id = sp.unk_id()

    # --------------------------------------------------------
    # Aggregate statistics
    # --------------------------------------------------------

    total_records = 0
    total_words = 0
    total_tokens = 0
    total_unk = 0
    total_chars = 0
    records_with_unk = 0

    token_counter = Counter()

    examples = []

    # --------------------------------------------------------
    # Stream JSONL directly
    # --------------------------------------------------------

    for text in tqdm(
        jsonl_text_iterator(dataset_path),
        desc=model_path.stem,
        unit=" records",
    ):

        # ----------------------------------------------------
        # Word count
        # ----------------------------------------------------

        words = text.split()

        if not words:
            continue

        # ----------------------------------------------------
        # SentencePiece tokenization
        # ----------------------------------------------------

        token_ids = sp.encode(
            text,
            out_type=int,
        )

        pieces = sp.id_to_piece(
            token_ids
        )

        num_tokens = len(
            token_ids
        )

        # ----------------------------------------------------
        # UNK statistics
        # ----------------------------------------------------

        num_unk = sum(
            token_id == unk_id
            for token_id in token_ids
        )

        if num_unk > 0:
            records_with_unk += 1

        # ----------------------------------------------------
        # Character count
        #
        # Keep this consistent with your previous evaluator:
        # exclude ordinary spaces.
        # ----------------------------------------------------

        clean_chars = len(
            text.replace(" ", "")
        )

        # ----------------------------------------------------
        # Aggregate
        # ----------------------------------------------------

        total_records += 1
        total_words += len(words)
        total_tokens += num_tokens
        total_unk += num_unk
        total_chars += clean_chars

        # ----------------------------------------------------
        # Token frequencies
        # ----------------------------------------------------

        token_counter.update(
            pieces
        )

        # ----------------------------------------------------
        # Examples
        # ----------------------------------------------------

        if len(examples) < num_examples:

            examples.append(
                {
                    "text": text,
                    "words": len(words),
                    "pieces": pieces,
                    "token_ids": token_ids,
                }
            )

    # ========================================================
    # DERIVED METRICS
    # ========================================================

    fertility = (
        total_tokens / total_words
        if total_words
        else 0.0
    )

    unk_rate = (
        total_unk / total_tokens
        if total_tokens
        else 0.0
    )

    avg_chars_per_token = (
        total_chars / total_tokens
        if total_tokens
        else 0.0
    )

    unk_record_rate = (
        records_with_unk / total_records
        if total_records
        else 0.0
    )

    # --------------------------------------------------------
    # Top tokens
    # --------------------------------------------------------

    top_tokens = []

    for token, count in token_counter.most_common(
        top_k
    ):

        frequency = (
            count / total_tokens
            if total_tokens
            else 0.0
        )

        top_tokens.append(
            {
                "token": token,
                "count": count,
                "frequency": frequency,
            }
        )

    return {
        "records": total_records,
        "words": total_words,
        "characters": total_chars,
        "tokens": total_tokens,
        "unique_tokens": len(token_counter),
        "unk_tokens": total_unk,
        "records_with_unk": records_with_unk,
        "fertility": fertility,
        "unk_rate": unk_rate,
        "unk_record_rate": unk_record_rate,
        "avg_chars_per_token": avg_chars_per_token,
        "token_counter": token_counter,
        "top_tokens": top_tokens,
        "examples": examples,
    }


# ============================================================
# FIND MODELS
# ============================================================

def find_tokenizers(
    model_dir: Path,
):

    models = sorted(
        model_dir.glob(
            "unigram_*.model"
        ),
        key=lambda p: int(
            p.stem.split("_")[-1]
        ),
    )

    if not models:

        raise FileNotFoundError(
            "No unigram_*.model files found "
            f"in {model_dir}"
        )

    return models


# ============================================================
# EVALUATE DATASET
# ============================================================

def evaluate_dataset(
    models,
    dataset_name,
    dataset_path,
    top_k,
    num_examples,
):

    print()
    print("=" * 80)
    print(
        f"EVALUATING {dataset_name.upper()}"
    )
    print("=" * 80)

    print(
        f"Dataset: {dataset_path}"
    )

    results = []

    for model_path in models:

        vocab_size = int(
            model_path.stem.split("_")[-1]
        )

        print()
        print(
            f"Model: {model_path.name}"
        )

        metrics = evaluate_tokenizer(
            model_path,
            dataset_path,
            top_k,
            num_examples,
        )

        result = {
            "dataset": dataset_name,
            "vocab_size": vocab_size,
            **metrics,
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print()
        print(
            f"  Records:             "
            f"{metrics['records']:,}"
        )

        print(
            f"  Words:               "
            f"{metrics['words']:,}"
        )

        print(
            f"  Characters:          "
            f"{metrics['characters']:,}"
        )

        print(
            f"  Tokens:              "
            f"{metrics['tokens']:,}"
        )

        print(
            f"  Unique tokens:       "
            f"{metrics['unique_tokens']:,}"
        )

        print(
            f"  Avg chars/token:     "
            f"{metrics['avg_chars_per_token']:.4f}"
        )

        print(
            f"  Fertility:           "
            f"{metrics['fertility']:.6f}"
        )

        print(
            f"  UNK tokens:          "
            f"{metrics['unk_tokens']:,}"
        )

        print(
            f"  UNK rate:            "
            f"{metrics['unk_rate']:.8%}"
        )

        print(
            f"  Records with UNK:    "
            f"{metrics['records_with_unk']:,}"
        )

        print(
            f"  UNK record rate:     "
            f"{metrics['unk_record_rate']:.8%}"
        )

        # ----------------------------------------------------
        # Top tokens
        # ----------------------------------------------------

        print()
        print(
            f"  TOP {top_k} TOKENS"
        )

        print(
            f"  {'Token':<25}"
            f"{'Count':>15}"
            f"{'Frequency':>15}"
        )

        print(
            "  " + "-" * 55
        )

        for item in metrics["top_tokens"]:

            token_display = (
                item["token"]
                .replace("\n", "\\n")
                .replace("\t", "\\t")
            )

            print(
                f"  {token_display:<25}"
                f"{item['count']:>15,}"
                f"{item['frequency']:>14.6%}"
            )

        # ----------------------------------------------------
        # Tokenization examples
        # ----------------------------------------------------

        print()
        print(
            "  TOKENIZATION EXAMPLES"
        )

        for i, example in enumerate(
            metrics["examples"],
            start=1,
        ):

            print()
            print(
                f"  Example {i}"
            )

            print(
                f"    Text: "
                f"{example['text'][:300]}"
            )

            print(
                f"    Words: "
                f"{example['words']}"
            )

            print(
                "    Pieces:"
            )

            print(
                "      "
                + " | ".join(
                    example["pieces"]
                )
            )

            print(
                "    IDs:"
            )

            print(
                "      "
                + " ".join(
                    str(x)
                    for x in example["token_ids"]
                )
            )

    return results


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

def save_results(
    results,
    output_path: Path,
):

    fieldnames = [
        "dataset",
        "vocab_size",
        "records",
        "words",
        "characters",
        "tokens",
        "unique_tokens",
        "unk_tokens",
        "records_with_unk",
        "fertility",
        "unk_rate",
        "unk_record_rate",
        "avg_chars_per_token",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                {
                    key: result[key]
                    for key in fieldnames
                }
            )

    print()
    print(
        f"Results written to: "
        f"{output_path}"
    )


# ============================================================
# SAVE TOKEN FREQUENCIES
# ============================================================

def save_token_frequencies(
    results,
    output_dir: Path,
):

    for result in results:

        dataset = result["dataset"]
        vocab_size = result["vocab_size"]

        output_path = (
            output_dir
            / (
                f"{dataset}_"
                f"unigram_{vocab_size}_"
                f"token_frequencies.csv"
            )
        )

        counter = result[
            "token_counter"
        ]

        total_tokens = result[
            "tokens"
        ]

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "token",
                    "count",
                    "frequency",
                ]
            )

            for token, count in (
                counter.most_common()
            ):

                frequency = (
                    count / total_tokens
                    if total_tokens
                    else 0.0
                )

                writer.writerow(
                    [
                        token,
                        count,
                        frequency,
                    ]
                )

        print(
            f"Saved token frequencies: "
            f"{output_path}"
        )


# ============================================================
# SAVE TOKENIZATION EXAMPLES
# ============================================================

def save_examples(
    results,
    output_dir: Path,
):

    for result in results:

        dataset = result["dataset"]
        vocab_size = result["vocab_size"]

        output_path = (
            output_dir
            / (
                f"{dataset}_"
                f"unigram_{vocab_size}_"
                f"examples.txt"
            )
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            for i, example in enumerate(
                result["examples"],
                start=1,
            ):

                f.write(
                    "=" * 80
                )
                f.write("\n")

                f.write(
                    f"Example {i}\n"
                )

                f.write(
                    f"Text:\n"
                    f"{example['text']}\n\n"
                )

                f.write(
                    f"Words: "
                    f"{example['words']}\n"
                )

                f.write(
                    "Pieces:\n"
                )

                f.write(
                    " | ".join(
                        example["pieces"]
                    )
                )

                f.write("\n\n")

                f.write(
                    "Token IDs:\n"
                )

                f.write(
                    " ".join(
                        str(x)
                        for x in example["token_ids"]
                    )
                )

                f.write("\n\n")

        print(
            f"Saved examples: "
            f"{output_path}"
        )


# ============================================================
# PLOT METRIC
# ============================================================

def plot_metric(
    results,
    metric,
    ylabel,
    title,
    output_path,
    multiply=1.0,
):

    plt.figure(
        figsize=(9, 6)
    )

    for dataset_name in [
        "validation",
        "test",
    ]:

        data = sorted(
            (
                r
                for r in results
                if r["dataset"]
                == dataset_name
            ),
            key=lambda r: r["vocab_size"],
        )

        if not data:
            continue

        vocab_sizes = [
            r["vocab_size"]
            for r in data
        ]

        values = [
            r[metric] * multiply
            for r in data
        ]

        plt.plot(
            vocab_sizes,
            values,
            marker="o",
            label=dataset_name.capitalize(),
        )

    plt.xlabel(
        "Vocabulary size"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved plot: {output_path}"
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(results):

    print()
    print("=" * 125)
    print("SUMMARY")
    print("=" * 125)

    print(
        f"{'Dataset':<12}"
        f"{'Vocab':>9}"
        f"{'Words':>14}"
        f"{'Tokens':>14}"
        f"{'UNK':>10}"
        f"{'Fertility':>12}"
        f"{'Chars/Tok':>12}"
        f"{'UNK Rate':>13}"
        f"{'UNK Records':>14}"
    )

    print("-" * 125)

    for r in sorted(
        results,
        key=lambda x: (
            x["dataset"],
            x["vocab_size"],
        ),
    ):

        print(
            f"{r['dataset']:<12}"
            f"{r['vocab_size']:>9,}"
            f"{r['words']:>14,}"
            f"{r['tokens']:>14,}"
            f"{r['unk_tokens']:>10,}"
            f"{r['fertility']:>12.6f}"
            f"{r['avg_chars_per_token']:>12.4f}"
            f"{r['unk_rate']:>12.6%}"
            f"{r['records_with_unk']:>14,}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SentencePiece Unigram models "
            "directly on JSONL held-out data."
        )
    )

    parser.add_argument(
        "--models",
        type=Path,
        default=Path("models"),
        help=(
            "Directory containing "
            "unigram_*.model files."
        ),
    )

    parser.add_argument(
        "--validation",
        required=True,
        type=Path,
        help="Validation JSONL file.",
    )

    parser.add_argument(
        "--test",
        required=True,
        type=Path,
        help="Test JSONL file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation"),
        help="Directory for statistics and plots.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of most frequent tokens to show.",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Number of examples per model/dataset.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not args.models.exists():
        raise FileNotFoundError(
            f"Model directory does not exist: "
            f"{args.models}"
        )

    if not args.validation.exists():
        raise FileNotFoundError(
            f"Validation file does not exist: "
            f"{args.validation}"
        )

    if not args.test.exists():
        raise FileNotFoundError(
            f"Test file does not exist: "
            f"{args.test}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = find_tokenizers(
        args.models
    )

    print()
    print("=" * 80)
    print("SENTENCEPIECE TOKENIZER EVALUATION")
    print("=" * 80)

    print(
        f"Found {len(models)} model(s):"
    )

    for model in models:
        print(
            f"  {model.name}"
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_results = evaluate_dataset(
        models,
        "validation",
        args.validation,
        args.top_k,
        args.examples,
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_results = evaluate_dataset(
        models,
        "test",
        args.test,
        args.top_k,
        args.examples,
    )

    results = (
        validation_results
        + test_results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results,
        args.output_dir
        / "tokenizer_results.csv",
    )

    save_token_frequencies(
        results,
        args.output_dir,
    )

    save_examples(
        results,
        args.output_dir,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_metric(
        results,
        metric="fertility",
        ylabel="Fertility (tokens / word)",
        title=(
            "SentencePiece Unigram "
            "Fertility vs Vocabulary Size"
        ),
        output_path=(
            args.output_dir
            / "fertility_vs_vocab.png"
        ),
    )

    plot_metric(
        results,
        metric="unk_rate",
        ylabel="Unknown-token rate (%)",
        title=(
            "SentencePiece Unigram "
            "Unknown-token Rate vs Vocabulary Size"
        ),
        output_path=(
            args.output_dir
            / "unk_rate_vs_vocab.png"
        ),
        multiply=100,
    )

    plot_metric(
        results,
        metric="avg_chars_per_token",
        ylabel="Average characters per token",
        title=(
            "Average Characters per Token "
            "vs Vocabulary Size"
        ),
        output_path=(
            args.output_dir
            / "chars_per_token_vs_vocab.png"
        ),
    )

    print_summary(
        results
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()