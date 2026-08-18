import json
import os
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

# ---- config ----
LANG_SPLIT = "hin"                  # Sangraha's split name for Nepali
TARGET_WORDS = 500_000_000
OUTPUT_DIR = "sangraha_hindi_sample"

# Reads from the HF_TOKEN environment variable by default. Set it with:
#   export HF_TOKEN=hf_xxxxxxxxxxxx        (Linux/Mac)
#   $env:HF_TOKEN="hf_xxxxxxxxxxxx"        (Windows PowerShell)
# A token isn't required for Sangraha specifically since it's a public
# dataset, but it's still worth setting -- it authenticates your requests
# and gets you the higher, logged-in rate limit instead of the anonymous one.
HF_TOKEN = os.environ.get("HF_TOKEN")

# Rough allocation across tiers. Verified is real, human-checked text with
# source-type metadata; unverified is filtered web text with no type field;
# synthetic is machine-translated Wikipedia -- included at a small share here
# only if you want extra volume/diversity, not as a primary source.
TIER_WORD_TARGETS = {
    "verified": int(TARGET_WORDS * 1.00),
    # "unverified": int(TARGET_WORDS * 0.25),
    # "synthetic": 0,   # left out by default -- uncomment and give it a share if you want it
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def word_count(text: str) -> int:
    return len(text.split())


def sample_tier(tier: str, target_words: int):
    if target_words <= 0:
        return

    ds = load_dataset(
        "ai4bharat/sangraha",
        data_dir=f"{tier}/{LANG_SPLIT}",
        streaming=True,
        split="train",
        token=HF_TOKEN,
    )

    out_path = os.path.join(OUTPUT_DIR, f"{tier}.jsonl")
    type_counts = defaultdict(int)
    total_words = 0

    with open(out_path, "w", encoding="utf-8") as out:
        with tqdm(total=target_words, unit="word", unit_scale=True, desc=f"[{tier}]") as pbar:
            for row in ds:
                text = row["text"]
                wc = word_count(text)
                if wc == 0:
                    continue

                record = {"doc_id": row.get("doc_id"), "text": text, "tier": tier}
                doc_type = row.get("type")  # only present in the "verified" tier
                if doc_type:
                    record["type"] = doc_type
                    type_counts[doc_type] += wc

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_words += wc
                pbar.update(wc)

                if total_words >= target_words:
                    break

    print(f"[{tier}] wrote {total_words:,} words to {out_path}")
    if type_counts:
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {t}: {c:,} words")


def main():
    if HF_TOKEN is None:
        print("Note: HF_TOKEN not set in environment -- proceeding with anonymous access.")
    for tier, target in TIER_WORD_TARGETS.items():
        sample_tier(tier, target)


if __name__ == "__main__":
    main()
