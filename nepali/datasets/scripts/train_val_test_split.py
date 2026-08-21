#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

TOKENS_PER_WORD = 1.4
TARGET_TRAIN_TOKENS = 500_000_000

MANUAL_RATIO = 0.20
SANGRAHA_RATIO = 0.80


# ============================================================
# WORD COUNT
# ============================================================

def count_words(text):
    return len(text.split())


# ============================================================
# READ + SEPARATE BY SOURCE
# ============================================================

def load_records(path):
    manual = []
    sangraha = []

    total_words = 0
    manual_words = 0
    sangraha_words = 0

    print(f"Reading {path}...")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: invalid JSON at line {line_no}")
                continue

            text = obj.get("text", "")

            if not isinstance(text, str):
                text = str(text)

            words = count_words(text)

            source = obj.get("source", "").lower()

            if source == "manual":
                manual.append((line, words))
                manual_words += words

            elif source in ("sangraha", "sangraha_nepali"):
                sangraha.append((line, words))
                sangraha_words += words

            else:
                raise ValueError(
                    f"Unknown source at line {line_no}: {source!r}"
                )

            total_words += words

            if line_no % 100_000 == 0:
                print(
                    f"\rProcessed {line_no:,} records | "
                    f"{total_words:,} words",
                    end="",
                )

    print()

    return (
        manual,
        sangraha,
        total_words,
        manual_words,
        sangraha_words,
    )


# ============================================================
# SHUFFLE
# ============================================================

def shuffle_records(records, seed):
    rng = random.Random(seed)
    rng.shuffle(records)


# ============================================================
# TAKE RECORDS UNTIL TARGET WORD COUNT
# ============================================================

def take_records(records, target_words, start_index):
    """
    Take complete documents until we reach approximately
    target_words.

    Returns:
        selected_records,
        new_index,
        actual_words
    """

    selected = []
    words = 0
    index = start_index

    while index < len(records) and words < target_words:

        record, record_words = records[index]

        selected.append(record)

        words += record_words
        index += 1

    return selected, index, words


# ============================================================
# SPLIT
# ============================================================

def split_source_records(
    records,
    total_words,
    train_ratio,
    val_ratio,
    test_ratio,
):
    """
    Split one source according to the requested split ratios.
    """

    train_target = total_words * train_ratio
    val_target = total_words * val_ratio
    test_target = total_words * test_ratio

    train, index, train_words = take_records(
        records,
        train_target,
        0,
    )

    val, index, val_words = take_records(
        records,
        val_target,
        index,
    )

    test, index, test_words = take_records(
        records,
        test_target,
        index,
    )

    return (
        train,
        val,
        test,
        train_words,
        val_words,
        test_words,
    )


# ============================================================
# WRITE
# ============================================================

def write_records(path, records):
    with path.open("w", encoding="utf-8") as f:
        for line in records:
            f.write(line)
            f.write("\n")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        default="splits",
        type=Path,
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Read corpus
    # --------------------------------------------------------

    (
        manual,
        sangraha,
        total_words,
        manual_words,
        sangraha_words,
    ) = load_records(args.input)

    print()
    print("=" * 70)
    print("CORPUS")
    print("=" * 70)

    print(f"Total words:     {total_words:,}")
    print(f"Manual words:    {manual_words:,}")
    print(f"Sangraha words:  {sangraha_words:,}")

    print(
        f"Manual ratio:    "
        f"{manual_words / total_words * 100:.3f}%"
    )

    print(
        f"Sangraha ratio:  "
        f"{sangraha_words / total_words * 100:.3f}%"
    )

    # --------------------------------------------------------
    # 2. Shuffle each source independently
    # --------------------------------------------------------

    print()
    print("Shuffling manual corpus...")
    shuffle_records(manual, args.seed)

    print("Shuffling Sangraha corpus...")
    shuffle_records(sangraha, args.seed + 1)

    # --------------------------------------------------------
    # 3. Calculate desired split sizes
    # --------------------------------------------------------

    target_train_words = (
        TARGET_TRAIN_TOKENS / TOKENS_PER_WORD
    )

    remaining_words = total_words - target_train_words

    target_val_words = remaining_words / 2
    target_test_words = remaining_words / 2

    total_target = (
        target_train_words
        + target_val_words
        + target_test_words
    )

    train_ratio = target_train_words / total_target
    val_ratio = target_val_words / total_target
    test_ratio = target_test_words / total_target

    print()
    print("=" * 70)
    print("TARGET SPLIT")
    print("=" * 70)

    print(
        f"Train:       {target_train_words:,.0f} words "
        f"(~{TARGET_TRAIN_TOKENS:,} tokens)"
    )

    print(
        f"Validation:  {target_val_words:,.0f} words"
    )

    print(
        f"Test:        {target_test_words:,.0f} words"
    )

    # --------------------------------------------------------
    # 4. Split manual
    # --------------------------------------------------------

    (
        manual_train,
        manual_val,
        manual_test,
        manual_train_words,
        manual_val_words,
        manual_test_words,
    ) = split_source_records(
        manual,
        manual_words,
        train_ratio,
        val_ratio,
        test_ratio,
    )

    # --------------------------------------------------------
    # 5. Split Sangraha
    # --------------------------------------------------------

    (
        sangraha_train,
        sangraha_val,
        sangraha_test,
        sangraha_train_words,
        sangraha_val_words,
        sangraha_test_words,
    ) = split_source_records(
        sangraha,
        sangraha_words,
        train_ratio,
        val_ratio,
        test_ratio,
    )

    # --------------------------------------------------------
    # 6. Combine sources within each split
    # --------------------------------------------------------

    train = manual_train + sangraha_train
    val = manual_val + sangraha_val
    test = manual_test + sangraha_test

    # Shuffle again so that manual/sangraha aren't grouped.
    shuffle_records(train, args.seed + 10)
    shuffle_records(val, args.seed + 11)
    shuffle_records(test, args.seed + 12)

    # --------------------------------------------------------
    # 7. Write
    # --------------------------------------------------------

    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "validation.jsonl"
    test_path = args.output_dir / "test.jsonl"

    print()
    print("Writing datasets...")

    write_records(train_path, train)
    write_records(val_path, val)
    write_records(test_path, test)

    # --------------------------------------------------------
    # 8. Statistics
    # --------------------------------------------------------

    train_words = (
        manual_train_words
        + sangraha_train_words
    )

    val_words = (
        manual_val_words
        + sangraha_val_words
    )

    test_words = (
        manual_test_words
        + sangraha_test_words
    )

    print()
    print("=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    print()

    print("TRAIN")
    print(f"  Total words:     {train_words:,}")
    print(f"  Estimated tokens:{train_words * TOKENS_PER_WORD:,.0f}")
    print(f"  Manual:          {manual_train_words:,}")
    print(f"  Sangraha:        {sangraha_train_words:,}")
    print(
        f"  Manual ratio:    "
        f"{manual_train_words / train_words * 100:.3f}%"
    )

    print()

    print("VALIDATION")
    print(f"  Total words:     {val_words:,}")
    print(f"  Estimated tokens:{val_words * TOKENS_PER_WORD:,.0f}")
    print(f"  Manual:          {manual_val_words:,}")
    print(f"  Sangraha:        {sangraha_val_words:,}")
    print(
        f"  Manual ratio:    "
        f"{manual_val_words / val_words * 100:.3f}%"
    )

    print()

    print("TEST")
    print(f"  Total words:     {test_words:,}")
    print(f"  Estimated tokens:{test_words * TOKENS_PER_WORD:,.0f}")
    print(f"  Manual:          {manual_test_words:,}")
    print(f"  Sangraha:        {sangraha_test_words:,}")
    print(
        f"  Manual ratio:    "
        f"{manual_test_words / test_words * 100:.3f}%"
    )

    print()
    print("=" * 70)
    print("FILES")
    print("=" * 70)

    print(train_path)
    print(val_path)
    print(test_path)


if __name__ == "__main__":
    main()