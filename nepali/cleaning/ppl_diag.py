#!/usr/bin/env python3

import argparse
import json
import math
import random
from pathlib import Path

import kenlm


BUCKETS = [
    ("<50", 0, 50),
    ("50-100", 50, 100),
    ("100-250", 100, 250),
    ("250-500", 250, 500),
    ("500-1000", 500, 1000),
    ("1000-5000", 1000, 5000),
    ("5000-10000", 5000, 10000),
    ("10000-100000", 10000, 100000),
    (">100000", 100000, float("inf")),
]


def calculate_perplexity(model, text):
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


def get_bucket(ppl):
    for name, low, high in BUCKETS:
        if low <= ppl < high:
            return name

    return None


def main():

    parser = argparse.ArgumentParser(
        description="Inspect documents at different KenLM perplexity ranges."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input cleaned JSONL file.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="KenLM .binary model.",
    )

    parser.add_argument(
        "--output",
        default="ppl_samples.jsonl",
        help="Output JSONL containing sampled documents.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of documents to sample per bucket.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    model_path = Path(args.model)
    output_path = Path(args.output)

    print(f"Loading model: {model_path}")
    model = kenlm.Model(str(model_path))

    # ---------------------------------------------------------
    # Store reservoir samples for each bucket
    # ---------------------------------------------------------

    samples = {
        bucket_name: []
        for bucket_name, _, _ in BUCKETS
    }

    bucket_counts = {
        bucket_name: 0
        for bucket_name, _, _ in BUCKETS
    }

    # Global statistics
    total = 0
    invalid = 0

    min_record = None
    max_record = None

    all_ppls = []

    # ---------------------------------------------------------
    # Reservoir sampling
    # ---------------------------------------------------------

    with input_path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue

            text = record.get("text")

            if not isinstance(text, str) or not text.strip():
                invalid += 1
                continue

            ppl = calculate_perplexity(
                model,
                text,
            )

            if ppl is None or not math.isfinite(ppl):
                invalid += 1
                continue

            total += 1
            all_ppls.append(ppl)

            # -------------------------------------------------
            # Track minimum
            # -------------------------------------------------

            if (
                min_record is None
                or ppl < min_record["perplexity"]
            ):
                min_record = {
                    "perplexity": ppl,
                    "text": text,
                    "line": line_number,
                }

            # -------------------------------------------------
            # Track maximum
            # -------------------------------------------------

            if (
                max_record is None
                or ppl > max_record["perplexity"]
            ):
                max_record = {
                    "perplexity": ppl,
                    "text": text,
                    "line": line_number,
                }

            # -------------------------------------------------
            # Bucket
            # -------------------------------------------------

            bucket = get_bucket(ppl)

            if bucket is None:
                continue

            bucket_counts[bucket] += 1

            bucket_samples = samples[bucket]

            # Reservoir sampling
            if len(bucket_samples) < args.samples:

                bucket_samples.append(
                    {
                        "perplexity": ppl,
                        "text": text,
                        "line": line_number,
                    }
                )

            else:

                index = random.randint(
                    0,
                    bucket_counts[bucket] - 1,
                )

                if index < args.samples:
                    bucket_samples[index] = {
                        "perplexity": ppl,
                        "text": text,
                        "line": line_number,
                    }

            if total % 10_000 == 0:
                print(
                    f"Scored {total:,} documents...",
                    flush=True,
                )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    all_ppls.sort()

    def percentile(p):
        if not all_ppls:
            return None

        index = (len(all_ppls) - 1) * p / 100

        lower = math.floor(index)
        upper = math.ceil(index)

        if lower == upper:
            return all_ppls[lower]

        fraction = index - lower

        return (
            all_ppls[lower]
            + fraction * (
                all_ppls[upper]
                - all_ppls[lower]
            )
        )

    print()
    print("=" * 70)
    print("PERPLEXITY DISTRIBUTION")
    print("=" * 70)

    print(f"Documents scored : {total:,}")
    print(f"Invalid/skipped  : {invalid:,}")

    print()

    for p in [
        0.1,
        0.5,
        1,
        2,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        97,
        98,
        99,
        99.5,
        99.9,
        99.95,
        99.99,
    ]:

        print(
            f"P{p:<6} = {percentile(p):,.4f}"
        )

    # ---------------------------------------------------------
    # Bucket counts
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("PPL BUCKETS")
    print("=" * 70)

    for bucket_name, _, _ in BUCKETS:

        count = bucket_counts[bucket_name]

        percentage = (
            100 * count / total
            if total
            else 0
        )

        print(
            f"{bucket_name:<15}"
            f"{count:>10,}"
            f"  ({percentage:>7.3f}%)"
        )

    # ---------------------------------------------------------
    # Extreme documents
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("LOWEST PERPLEXITY DOCUMENT")
    print("=" * 70)

    print(
        f"PPL: {min_record['perplexity']:.4f}"
    )
    print(
        f"Line: {min_record['line']}"
    )
    print(
        min_record["text"]
    )

    print()
    print("=" * 70)
    print("HIGHEST PERPLEXITY DOCUMENT")
    print("=" * 70)

    print(
        f"PPL: {max_record['perplexity']:.4f}"
    )
    print(
        f"Line: {max_record['line']}"
    )
    print(
        max_record["text"]
    )

    # ---------------------------------------------------------
    # Save samples
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as out:

        for bucket_name, _, _ in BUCKETS:

            bucket_samples = samples[bucket_name]

            # Sort samples by PPL
            bucket_samples.sort(
                key=lambda x: x["perplexity"]
            )

            for sample in bucket_samples:

                output_record = {
                    "bucket": bucket_name,
                    "perplexity": sample["perplexity"],
                    "input_line": sample["line"],
                    "text": sample["text"],
                }

                out.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print()
    print("=" * 70)
    print(f"Saved samples to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()