#!/usr/bin/env python3

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


# ============================================================
# Worker-side functions
# ============================================================

def word_tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u0900-\u097F]+", text, flags=re.UNICODE)


def make_shingles(text: str, shingle_size: int) -> list[str]:
    words = word_tokenize(text)
    if len(words) < shingle_size:
        return []
    words = [word.lower() for word in words]
    return [" ".join(words[i:i + shingle_size]) for i in range(len(words) - shingle_size + 1)]


def create_minhash(text: str, num_perm: int, shingle_size: int) -> Optional[list[int]]:
    shingles = make_shingles(text, shingle_size)
    if not shingles:
        return None
    minhash = MinHash(num_perm=num_perm)
    for shingle in shingles:
        minhash.update(shingle.encode("utf-8"))
    return minhash.hashvalues.tolist()


def reconstruct_minhash(hashvalues: list[int]) -> MinHash:
    import numpy as np
    hashvalues = np.asarray(hashvalues, dtype=np.uint64)
    minhash = MinHash(num_perm=len(hashvalues))
    minhash.hashvalues = hashvalues
    return minhash


def process_chunk(item):
    """
    Worker function -- runs in a subprocess for BOTH the manual and Sangraha
    passes now, since both need the same (line_number, record, num_perm,
    shingle_size) -> (line_number, record, minhash_bytes, word_count) shape.
    """
    line_number, record, num_perm, shingle_size = item
    text = record.get("text")

    if not isinstance(text, str):
        return (line_number, None, None, 0)

    words = word_tokenize(text)
    if len(words) < shingle_size:
        return (line_number, None, None, len(words))

    minhash_bytes = create_minhash(text, num_perm, shingle_size)
    if minhash_bytes is None:
        return (line_number, None, None, len(words))

    return (line_number, record, minhash_bytes, len(words))


# ============================================================
# Main deduplicator
# ============================================================

class ManualPriorityDeduplicator:

    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        shingle_size: int = 5,
        workers: int = 8,
        manual_target_words: int = 100_000_000,
        target_words: int = 500_000_000,
    ):
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.workers = workers
        self.manual_target_words = manual_target_words
        self.target_words = target_words

        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

        self.manual_indexed = 0
        self.manual_words = 0

        self.sangraha_total = 0
        self.sangraha_kept = 0
        self.sangraha_duplicates = 0
        self.sangraha_skipped = 0
        self.sangraha_words_kept = 0

    # ========================================================
    # Manual corpus: index + write in ONE pass, capped at
    # manual_target_words. Minhash computation is parallelized
    # across worker processes; LSH insertion stays in the main
    # process since MinHashLSH isn't shareable across processes.
    # ========================================================

    def process_manual(self, manual_path: Path, output_path: Path):
        print(f"Processing manual corpus (cap: {self.manual_target_words:,} words): {manual_path}")

        def manual_records():
            with manual_path.open("r", encoding="utf-8") as src:
                for line_number, line in enumerate(src, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yield line_number, record

        def worker_input():
            for line_number, record in manual_records():
                yield (line_number, record, self.num_perm, self.shingle_size)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as dst:
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                results = executor.map(process_chunk, worker_input(), chunksize=64)

                for line_number, record, minhash_bytes, word_count in tqdm(
                    results, desc="Indexing manual", unit="rec"
                ):
                    if record is None or minhash_bytes is None:
                        continue

                    # stop BEFORE writing anything that would push us over
                    # the cap, unless we haven't kept anything yet (avoids
                    # getting stuck if a single doc exceeds the whole cap)
                    if self.manual_words + word_count > self.manual_target_words and self.manual_indexed > 0:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    minhash = reconstruct_minhash(minhash_bytes)
                    self.lsh.insert(f"manual:{line_number}", minhash)

                    record["source"] = "manual"
                    dst.write(json.dumps(record, ensure_ascii=False) + "\n")

                    self.manual_indexed += 1
                    self.manual_words += word_count

                    if self.manual_words >= self.manual_target_words:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

        print()
        print(f"Manual documents kept : {self.manual_indexed:,}")
        print(f"Manual words kept     : {self.manual_words:,}")

    # ========================================================
    # Read Sangraha records
    # ========================================================

    def read_sangraha(self, sangraha_path: Path):
        with sangraha_path.open("r", encoding="utf-8") as src:
            for line_number, line in enumerate(src, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    self.sangraha_skipped += 1
                    continue
                yield (line_number, record)

    # ========================================================
    # Process Sangraha: keep only docs that don't near-duplicate
    # anything in the manual-priority LSH index, until the overall
    # target_words is reached.
    # ========================================================

    def process_sangraha(self, sangraha_path: Path, output_path: Path):
        remaining_words = self.target_words - self.manual_words

        if remaining_words <= 0:
            print("Manual corpus already meets/exceeds the target word count.")
            return

        print()
        print(f"Target total words : {self.target_words:,}")
        print(f"Manual words        : {self.manual_words:,}")
        print(f"Sangraha needed     : {remaining_words:,}")

        records = self.read_sangraha(sangraha_path)

        def worker_input():
            for line_number, record in records:
                yield (line_number, record, self.num_perm, self.shingle_size)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("a", encoding="utf-8") as dst:
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                results = executor.map(process_chunk, worker_input(), chunksize=64)

                for line_number, record, minhash_bytes, word_count in tqdm(
                    results, desc="Deduplicating Sangraha", unit="rec"
                ):
                    self.sangraha_total += 1

                    if record is None or minhash_bytes is None:
                        self.sangraha_skipped += 1
                        continue

                    minhash = reconstruct_minhash(minhash_bytes)  # bug fix: was called with an extra arg

                    if self.lsh.query(minhash):
                        self.sangraha_duplicates += 1
                        continue

                    record["source"] = "sangraha"
                    dst.write(json.dumps(record, ensure_ascii=False) + "\n")

                    self.sangraha_kept += 1
                    self.sangraha_words_kept += word_count

                    if self.manual_words + self.sangraha_words_kept >= self.target_words:
                        print()
                        print("Reached target word count.")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

    # ========================================================
    # Complete merge
    # ========================================================

    def merge(self, manual_path: Path, sangraha_path: Path, output_path: Path):
        self.process_manual(manual_path, output_path)
        self.process_sangraha(sangraha_path, output_path)

        final_words = self.manual_words + self.sangraha_words_kept

        print()
        print("=" * 70)
        print("FINAL DATASET")
        print("=" * 70)
        print(f"Manual records        : {self.manual_indexed:,}")
        print(f"Manual words          : {self.manual_words:,}")
        print(f"Sangraha processed    : {self.sangraha_total:,}")
        print(f"Sangraha kept         : {self.sangraha_kept:,}")
        print(f"Sangraha duplicates   : {self.sangraha_duplicates:,}")
        print(f"Sangraha words kept   : {self.sangraha_words_kept:,}")
        print(f"FINAL WORDS           : {final_words:,}")
        print(f"TARGET WORDS          : {self.target_words:,}")
        print(f"Output                : {output_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parallel manual-priority MinHash deduplication for large Sangraha corpora."
    )
    parser.add_argument("--manual", required=True, help="Cleaned manual JSONL.")
    parser.add_argument("--sangraha", required=True, help="Cleaned Sangraha JSONL.")
    parser.add_argument("--output", required=True, help="Final merged JSONL.")
    parser.add_argument("--threshold", type=float, default=0.9, help="MinHash Jaccard threshold. Default: 0.9")
    parser.add_argument("--num-perm", type=int, default=128, help="MinHash permutations. Default: 128")
    parser.add_argument("--shingle-size", type=int, default=5, help="Words per shingle. Default: 5")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes. Default: 8")
    parser.add_argument(
        "--manual-target-words", type=int, default=100_000_000,
        help="Cap on manual corpus contribution, in words. Default: 100,000,000",
    )
    parser.add_argument(
        "--target-words", type=int, default=500_000_000,
        help="Target final (manual + sangraha) corpus size in words. Default: 500,000,000",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.manual_target_words > args.target_words:
        parser.error("--manual-target-words cannot exceed --target-words")

    deduplicator = ManualPriorityDeduplicator(
        threshold=args.threshold,
        num_perm=args.num_perm,
        shingle_size=args.shingle_size,
        workers=args.workers,
        manual_target_words=args.manual_target_words,
        target_words=args.target_words,
    )

    deduplicator.merge(
        manual_path=Path(args.manual),
        sangraha_path=Path(args.sangraha),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()