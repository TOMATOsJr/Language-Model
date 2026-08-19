#!/usr/bin/env python3

import argparse
import json
import math
import statistics
from pathlib import Path

import kenlm


def calculate_perplexity(model, text: str) -> float | None:
    """
    Calculate word-level perplexity using a KenLM model.

    KenLM returns log10 probabilities, so:

        PPL = 10 ^ (-score / N)

    where N is the number of whitespace-separated words.
    """

    if not text or not text.strip():
        return None

    words = text.split()

    if not words:
        return None

    try:
        score = model.score(
            text,
            bos=True,
            eos=True,
        )

        return 10 ** (-score / len(words))

    except Exception:
        return None


def percentile(values, p):
    """
    Calculate percentile using linear interpolation.
    """

    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * (p / 100.0)

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    fraction = position - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * fraction
    )


def score_documents(
    input_path: Path,
    model,
):
    """
    Read JSONL and calculate perplexity for every valid document.

    Returns:
        documents: list of (record, perplexity)
        invalid_count: number of records that could not be scored
    """

    documents = []
    invalid_count = 0

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_count += 1
                continue

            text = record.get("text")

            if not isinstance(text, str) or not text.strip():
                invalid_count += 1
                continue

            ppl = calculate_perplexity(
                model,
                text,
            )

            if ppl is None or not math.isfinite(ppl):
                invalid_count += 1
                continue

            documents.append(
                (record, ppl)
            )

            if len(documents) % 10_000 == 0:
                print(
                    f"Scored {len(documents):,} documents...",
                    flush=True,
                )

    return documents, invalid_count


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Score a cleaned Nepali JSONL corpus with a "
            "KenLM model and filter by perplexity."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input cleaned JSONL file.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to KenLM .binary model.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output filtered JSONL file.",
    )

    parser.add_argument(
        "--top-percent",
        type=float,
        default=0.0,
        help=(
            "Remove the highest-PPL X%% of documents. "
            "Default: 0."
        ),
    )

    parser.add_argument(
        "--bottom-percent",
        type=float,
        default=0.0,
        help=(
            "Remove the lowest-PPL X%% of documents. "
            "Default: 0."
        ),
    )

    parser.add_argument(
        "--min-perplexity",
        type=float,
        default=None,
        help=(
            "Remove documents whose perplexity is below "
            "this threshold."
        ),
    )

    parser.add_argument(
        "--max-perplexity",
        type=float,
        default=None,
        help=(
            "Remove documents whose perplexity is above "
            "this threshold."
        ),
    )

    parser.add_argument(
        "--stats-output",
        default=None,
        help=(
            "Optional JSON file containing perplexity "
            "statistics."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"KenLM model not found: {model_path}"
        )

    if not 0 <= args.top_percent < 100:
        raise ValueError(
            "--top-percent must be between 0 and 100."
        )

    if not 0 <= args.bottom_percent < 100:
        raise ValueError(
            "--bottom-percent must be between 0 and 100."
        )

    if args.max_perplexity is not None:
        if args.max_perplexity <= 0:
            raise ValueError(
                "--max-perplexity must be positive."
            )

    if args.min_perplexity is not None:
        if args.min_perplexity <= 0:
            raise ValueError(
                "--min-perplexity must be positive."
            )

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    print(f"Loading KenLM model: {model_path}")

    model = kenlm.Model(
        str(model_path)
    )

    # ---------------------------------------------------------
    # Score documents
    # ---------------------------------------------------------

    print(f"Reading: {input_path}")

    documents, invalid_count = score_documents(
        input_path,
        model,
    )

    if not documents:
        raise RuntimeError(
            "No documents could be scored."
        )

    perplexities = [
        ppl
        for _, ppl in documents
    ]

    # ---------------------------------------------------------
    # Basic statistics
    # ---------------------------------------------------------

    min_ppl = min(perplexities)
    max_ppl = max(perplexities)
    mean_ppl = statistics.mean(perplexities)
    median_ppl = statistics.median(perplexities)

    print()
    print("=" * 60)
    print("PERPLEXITY STATISTICS")
    print("=" * 60)

    print(
        f"Documents scored : {len(documents):,}"
    )

    print(
        f"Invalid/skipped  : {invalid_count:,}"
    )

    print(
        f"Minimum PPL      : {min_ppl:.4f}"
    )

    print(
        f"Maximum PPL      : {max_ppl:.4f}"
    )

    print(
        f"Mean PPL         : {mean_ppl:.4f}"
    )

    print(
        f"Median PPL       : {median_ppl:.4f}"
    )

    print()
    print("Percentiles:")

    for p in [
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
        99.5,
        99.9,
    ]:

        value = percentile(
            perplexities,
            p,
        )

        print(
            f"P{p:<5} = {value:.4f}"
        )

    # ---------------------------------------------------------
    # Determine percentile thresholds
    # ---------------------------------------------------------

    bottom_threshold = None
    top_threshold = None

    if args.bottom_percent > 0:
        bottom_threshold = percentile(
            perplexities,
            args.bottom_percent,
        )

    if args.top_percent > 0:
        top_threshold = percentile(
            perplexities,
            100.0 - args.top_percent,
        )

    print()
    print("=" * 60)
    print("FILTER CONFIGURATION")
    print("=" * 60)

    if bottom_threshold is not None:
        print(
            f"Bottom {args.bottom_percent}%:"
            f" PPL <= {bottom_threshold:.4f}"
        )
    else:
        print("Bottom percentile filtering: disabled")

    if top_threshold is not None:
        print(
            f"Top {args.top_percent}%:"
            f" PPL >= {top_threshold:.4f}"
        )
    else:
        print("Top percentile filtering: disabled")

    if args.max_perplexity is not None:
        print(
            f"Maximum PPL threshold:"
            f" {args.max_perplexity:.4f}"
        )
    else:
        print("Maximum PPL threshold: disabled")

    if args.min_perplexity is not None:
        print(
            f"Minimum PPL threshold:"
            f" {args.min_perplexity:.4f}"
        )
    else:
        print("Minimum PPL threshold: disabled")

    # ---------------------------------------------------------
    # Filtering
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    kept = 0
    removed_bottom = 0
    removed_top = 0
    removed_threshold = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as out:

        for record, ppl in documents:

            # Bottom percentile
            if (
                bottom_threshold is not None
                and ppl <= bottom_threshold
            ):
                removed_bottom += 1
                continue

            # Top percentile
            if (
                top_threshold is not None
                and ppl >= top_threshold
            ):
                removed_top += 1
                continue

            # Absolute threshold
            if (
                args.min_perplexity is not None
                and ppl < args.min_perplexity
            ):
                removed_threshold += 1
                continue

            if (
                args.max_perplexity is not None
                and ppl > args.max_perplexity
            ):
                removed_threshold += 1
                continue

            out.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            kept += 1

    # ---------------------------------------------------------
    # Final statistics
    # ---------------------------------------------------------

    removed_total = (
        removed_bottom
        + removed_top
        + removed_threshold
    )

    print()
    print("=" * 60)
    print("FILTER RESULTS")
    print("=" * 60)

    print(
        f"Input scored      : {len(documents):,}"
    )

    print(
        f"Removed bottom    : {removed_bottom:,}"
    )

    print(
        f"Removed top       : {removed_top:,}"
    )

    print(
        f"Removed threshold : {removed_threshold:,}"
    )

    print(
        f"Total removed     : {removed_total:,}"
    )

    print(
        f"Kept              : {kept:,}"
    )

    print(
        f"Output             : {output_path}"
    )

    # ---------------------------------------------------------
    # Optional statistics JSON
    # ---------------------------------------------------------

    if args.stats_output:

        stats = {
            "input": str(input_path),
            "model": str(model_path),
            "output": str(output_path),
            "documents_scored": len(documents),
            "invalid_or_skipped": invalid_count,
            "minimum_perplexity": min_ppl,
            "maximum_perplexity": max_ppl,
            "mean_perplexity": mean_ppl,
            "median_perplexity": median_ppl,
            "percentiles": {
                str(p): percentile(
                    perplexities,
                    p,
                )
                for p in [
                    1,
                    5,
                    10,
                    25,
                    50,
                    75,
                    90,
                    95,
                    99,
                    99.5,
                    99.9,
                ]
            },
            "filter": {
                "top_percent": args.top_percent,
                "bottom_percent": args.bottom_percent,
                "max_perplexity": args.max_perplexity,
                "min_perplexity": args.min_perplexity,
                "top_threshold": top_threshold,
                "bottom_threshold": bottom_threshold,
            },
            "results": {
                "removed_bottom": removed_bottom,
                "removed_top": removed_top,
                "removed_threshold": removed_threshold,
                "total_removed": removed_total,
                "kept": kept,
            },
        }

        with open(
            args.stats_output,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                stats,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"Statistics: {args.stats_output}"
        )


if __name__ == "__main__":
    main()