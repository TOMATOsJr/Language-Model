import json

INPUT = "./datasets/cleaned_ppl.jsonl"

total_alpha = 0
devanagari_alpha = 0
total_docs = 0

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        record = json.loads(line)
        text = record.get("text", "")

        if not text:
            continue

        total_docs += 1

        for ch in text:
            if not ch.isalpha():
                continue

            total_alpha += 1

            if "\u0900" <= ch <= "\u097F":
                devanagari_alpha += 1

percentage = (
    100 * devanagari_alpha / total_alpha
    if total_alpha
    else 0.0
)

print(f"Documents:              {total_docs:,}")
print(f"Alphabetic characters:   {total_alpha:,}")
print(f"Devanagari characters:   {devanagari_alpha:,}")
print(f"Devanagari percentage:   {percentage:.2f}%")