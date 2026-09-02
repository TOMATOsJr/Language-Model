from pathlib import Path
import argparse

total_word_count = 0

parser = argparse.ArgumentParser(
    description="Count words in text files"
)

parser.add_argument(
    "-d",
    "--directory",
    type=str,
    default = "nepali_corpus",
    help="Directory containing text files"
)

args = parser.parse_args()

for file in Path(args.directory).rglob("*.txt"):
    word_count = len(file.read_text(encoding="utf-8").split())
    print(f"{file}: {word_count} words")
    total_word_count += word_count

print(f"Total word count: {total_word_count}")
