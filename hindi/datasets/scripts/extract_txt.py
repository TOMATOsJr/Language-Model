#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def jsonl_to_txt(
    input_path: Path,
    output_path: Path,
):
    """
    Extract the `text` field from every JSONL record
    and write it to a plain text file.

    Each JSONL record becomes one text block.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = 0
    characters = 0

    with (
        input_path.open(
            "r",
            encoding="utf-8",
        ) as infile,
        output_path.open(
            "w",
            encoding="utf-8",
        ) as outfile,
    ):

        for line in tqdm(
            infile,
            desc=f"Extracting {input_path.name}",
            unit=" records",
        ):

            line = line.strip()

            if not line:
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

            if not text:
                continue

            outfile.write(text)
            outfile.write("\n")

            records += 1
            characters += len(text)

    print(
        f"{input_path.name}: "
        f"{records:,} records, "
        f"{characters:,} characters"
    )

    print(
        f"Written to: {output_path}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract text fields from JSONL "
            "datasets into TXT files."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input JSONL file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output TXT file.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {args.input}"
        )

    jsonl_to_txt(
        args.input,
        args.output,
    )


if __name__ == "__main__":
    main()