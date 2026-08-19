import json

INPUT = "verified.jsonl"
OUTPUT = "sangraha_15L_ne.txt"
LIMIT = 150_000

written = 0

with open(INPUT, "r", encoding="utf-8") as src, \
     open(OUTPUT, "w", encoding="utf-8") as dst:

    for line in src:
        if written >= LIMIT:
            break

        line = line.strip()

        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = record.get("text")

        if not isinstance(text, str) or not text.strip():
            continue

        # Keep each document on one line for KenLM.
        text = " ".join(text.split())

        dst.write(text + "\n")
        written += 1

print(f"Wrote {written:,} documents to {OUTPUT}")