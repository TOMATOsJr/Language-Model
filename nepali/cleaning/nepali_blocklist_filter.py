#!/usr/bin/env python3

import argparse
import json
import re
import unicodedata
from pathlib import Path


class NepaliBlocklistFilter:
    def __init__(
        self,
        blocklist_path: str,
        max_blocked_ratio: float = 0.02,
    ):
        self.max_blocked_ratio = max_blocked_ratio
        self.blocklist = self._load_blocklist(blocklist_path)

        if not self.blocklist:
            raise ValueError("Blocklist is empty.")

        escaped = sorted(
            (re.escape(word) for word in self.blocklist),
            key=len,
            reverse=True,
        )

        self.pattern = re.compile(
            r"(?<!\w)(?:"
            + "|".join(escaped)
            + r")(?!\w)",
            flags=re.UNICODE,
        )

    @staticmethod
    def _load_blocklist(path: str) -> set[str]:
        words = set()

        with open(
            path,
            "r",
            encoding="utf-8-sig",
        ) as f:

            for line in f:
                word = line.strip()

                if not word:
                    continue

                if word.startswith("#"):
                    continue

                word = unicodedata.normalize(
                    "NFC",
                    word,
                )

                words.add(word)

        return words

    def blocked_word_count(self, text: str) -> int:
        """
        Return the number of blocked-word occurrences
        in the document.

        Repeated occurrences count separately.
        """

        return len(
            self.pattern.findall(text)
        )

    def blocked_word_ratio(self, text: str) -> float:
        """
        Return:

            blocked word occurrences / total words
        """

        total_words = len(
            re.findall(
                r"[\w\u0900-\u097F]+",
                text,
                flags=re.UNICODE,
            )
        )

        if total_words == 0:
            return 0.0

        blocked_count = self.blocked_word_count(text)

        return blocked_count / total_words

    def is_unsafe(self, text: str) -> bool:
        """
        Remove the document if blocked words make up
        at least max_blocked_ratio of all words.
        """

        ratio = self.blocked_word_ratio(text)

        return ratio >= self.max_blocked_ratio


def process_manual(
    input_path: str,
    output_path: str,
    blocklist_path: str,
    max_blocked_ratio: float,
):
    blocklist_filter = NepaliBlocklistFilter(
        blocklist_path=blocklist_path,
        max_blocked_ratio=max_blocked_ratio,
    )

    total = 0
    kept = 0
    removed = 0
    invalid = 0

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as src, open(
        output_path,
        "w",
        encoding="utf-8",
    ) as dst:

        for line_number, line in enumerate(
            src,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            total += 1

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                invalid += 1
                continue

            text = record.get("text")

            if (
                not isinstance(text, str)
                or not text.strip()
            ):
                invalid += 1
                continue

            # -------------------------------------------------
            # Safety filtering
            # -------------------------------------------------

            blocked_count = (
                blocklist_filter.blocked_word_count(text)
            )

            total_words = len(
                re.findall(
                    r"[\w\u0900-\u097F]+",
                    text,
                    flags=re.UNICODE,
                )
            )

            blocked_ratio = (
                blocked_count / total_words
                if total_words > 0
                else 0.0
            )

            if blocked_ratio >= max_blocked_ratio:
                removed += 1
                continue

            # -------------------------------------------------
            # Keep document
            # -------------------------------------------------

            record["source"] = "manual"

            dst.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            kept += 1

    print()
    print("=" * 60)
    print("NEPALI BLOCKLIST FILTER")
    print("=" * 60)

    print(f"Input records       : {total:,}")
    print(f"Invalid/skipped     : {invalid:,}")
    print(f"Blocked/removed     : {removed:,}")
    print(f"Kept                : {kept:,}")

    if total:
        print(
            f"Removed percentage  : "
            f"{removed / total * 100:.2f}%"
        )

    print(
        f"Blocklist words     : "
        f"{len(blocklist_filter.blocklist):,}"
    )

    print(
        f"Maximum blocked ratio: "
        f"{max_blocked_ratio:.2%}"
    )

    print(f"Output              : {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter a Nepali manual JSONL corpus "
            "using a blocklist density threshold."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input manual JSONL.",
    )

    parser.add_argument(
        "--blocklist",
        required=True,
        help="Blocklist file, one word/phrase per line.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Filtered manual JSONL.",
    )

    parser.add_argument(
        "--max-blocked-ratio",
        type=float,
        default=0.02,
        help=(
            "Remove a document when the proportion "
            "of blocked-word occurrences is at least "
            "this value. Default: 0.02 (2%%)."
        ),
    )

    args = parser.parse_args()

    if not 0.0 <= args.max_blocked_ratio <= 1.0:
        parser.error(
            "--max-blocked-ratio must be between 0 and 1"
        )

    process_manual(
        input_path=args.input,
        output_path=args.output,
        blocklist_path=args.blocklist,
        max_blocked_ratio=args.max_blocked_ratio,
    )


if __name__ == "__main__":
    main()