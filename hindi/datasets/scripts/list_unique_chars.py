#!/usr/bin/env python3

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

from tqdm import tqdm


def extract_unique_chars(
    input_path: Path,
):
    """
    Extract all unique Unicode characters from the `text`
    field of a JSONL dataset.
    """

    char_counts = Counter()
    total_chars = 0
    total_records = 0

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in tqdm(
            f,
            desc="Scanning dataset",
            unit=" lines",
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = record.get("text", "")

            if not isinstance(text, str):
                continue

            total_records += 1
            total_chars += len(text)

            char_counts.update(text)

    return (
        char_counts,
        total_records,
        total_chars,
    )


def write_results(
    char_counts: Counter,
    output_path: Path,
):

    # Sort by Unicode code point.
    chars = sorted(
        char_counts,
        key=ord,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "codepoint\tcharacter\tname\tcount\n"
        )

        for char in chars:

            codepoint = (
                f"U+{ord(char):04X}"
            )

            name = unicodedata.name(
                char,
                "UNKNOWN",
            )

            # Make whitespace visible.
            display_char = repr(char)[1:-1]

            f.write(
                f"{codepoint}\t"
                f"{display_char}\t"
                f"{name}\t"
                f"{char_counts[char]}\n"
            )


def print_summary(
    char_counts: Counter,
    total_records: int,
    total_chars: int,
):

    print()
    print("=" * 70)
    print("CHARACTER ANALYSIS")
    print("=" * 70)

    print(
        f"Records:          {total_records:,}"
    )

    print(
        f"Total characters: {total_chars:,}"
    )

    print(
        f"Unique characters:{len(char_counts):,}"
    )

    print()
    print("CHARACTERS")
    print("-" * 70)

    for char in sorted(
        char_counts,
        key=ord,
    ):

        codepoint = (
            f"U+{ord(char):04X}"
        )

        name = unicodedata.name(
            char,
            "UNKNOWN",
        )

        display_char = repr(char)[1:-1]

        print(
            f"{codepoint:<10}"
            f"{display_char:<8}"
            f"{char_counts[char]:>12,}"
            f"  {name}"
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract unique Unicode characters "
            "from a JSONL dataset."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input JSONL dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("unique_chars.tsv"),
        help="Output TSV file.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.input}"
        )

    char_counts, total_records, total_chars = (
        extract_unique_chars(args.input)
    )

    write_results(
        char_counts,
        args.output,
    )

    print_summary(
        char_counts,
        total_records,
        total_chars,
    )

    print()
    print(
        f"Results written to: {args.output}"
    )


if __name__ == "__main__":
    main()