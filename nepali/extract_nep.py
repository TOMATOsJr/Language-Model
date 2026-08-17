import csv
import gzip
import glob
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from threading import Lock

import trafilatura
from tqdm import tqdm
from huggingface_hub import HfFileSystem
from warcio.recordloader import ArcWarcRecordLoader

# ---- config ----
RECORDS_GLOB = "nepali_only.csv"      # your extracted index CSVs
OUTPUT_DIR = "nepali_corpus"                   # each record gets {id}.txt + {id}.json here
MAX_WORKERS = 32                           # network-bound; tune based on observed throughput/errors
HF_BUCKET_PREFIX = "buckets/commoncrawl/commoncrawl"
ID_WIDTH_HEADROOM = 2                      # extra digits of padding beyond current record count

fs = HfFileSystem()
pbar_lock = Lock()


def load_records(records_glob: str) -> tuple[dict, dict]:
    """Reads all index CSVs, then assigns each record a stable sequential id
    based on a deterministic sort (warc_filename, offset) -- NOT dispatch/
    completion order, which would vary run to run under concurrency.
    Returns (grouped-by-warc-file dict, id_width)."""
    all_entries = []
    for path in glob.glob(records_glob):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_entries.append(
                    {
                        "url": row["url"],
                        "warc_filename": row["warc_filename"],
                        "offset": int(row["warc_record_offset"]),
                        "length": int(row["warc_record_length"]),
                        "content_languages": row.get("content_languages", ""),
                    }
                )

    all_entries.sort(key=lambda e: (e["warc_filename"], e["offset"]))

    id_width = len(str(len(all_entries))) + ID_WIDTH_HEADROOM
    for i, entry in enumerate(all_entries):
        entry["id"] = str(i).zfill(id_width)

    grouped = defaultdict(list)
    for entry in all_entries:
        grouped[entry["warc_filename"]].append(entry)

    return grouped, id_width


def extract_one_record(hf_file, entry: dict) -> tuple[str, dict] | None:
    """Range-reads one WARC record, decompresses it, parses out the HTML,
    and runs trafilatura on it. Returns (text, metadata) or None on failure."""
    hf_file.seek(entry["offset"])
    raw = hf_file.read(entry["length"])

    decompressed = gzip.decompress(raw)

    loader = ArcWarcRecordLoader()
    record = loader.parse_record_stream(BytesIO(decompressed))
    html_bytes = record.content_stream().read()

    charset = "utf-8"
    content_type = record.http_headers.get_header("Content-Type") if record.http_headers else None
    if content_type and "charset=" in content_type:
        charset = content_type.split("charset=")[-1].strip()

    try:
        html = html_bytes.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        html = html_bytes.decode("utf-8", errors="replace")

    text = trafilatura.extract(html, url=entry["url"], favor_recall=True)
    if not text:
        return None

    metadata = {
        "id": entry["id"],
        "url": entry["url"],
        "warc_filename": entry["warc_filename"],
        "warc_record_offset": entry["offset"],
        "warc_record_length": entry["length"],
        "content_languages": entry["content_languages"],
        "char_count": len(text),
        "word_count": len(text.split()),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    return text, metadata


def process_warc_file(warc_filename: str, entries: list, pbar: tqdm) -> str:
    """Opens one remote WARC file once, extracts every record we need from it,
    and writes a {id}.txt + {id}.json pair per successful record."""
    hf_path = f"{HF_BUCKET_PREFIX}/{warc_filename}"
    written = 0
    skipped = 0
    failed = 0

    try:
        with fs.open(hf_path, "rb") as hf_file:
            for entry in entries:
                txt_path = os.path.join(OUTPUT_DIR, f"{entry['id']}.txt")
                json_path = os.path.join(OUTPUT_DIR, f"{entry['id']}.json")

                # resumability: skip records already extracted in a previous run
                if os.path.exists(txt_path) and os.path.exists(json_path):
                    skipped += 1
                    with pbar_lock:
                        pbar.update(1)
                    continue

                try:
                    result = extract_one_record(hf_file, entry)
                except Exception:
                    failed += 1
                    with pbar_lock:
                        pbar.update(1)
                    continue

                if result is None:
                    failed += 1
                    with pbar_lock:
                        pbar.update(1)
                    continue

                text, metadata = result

                # each record writes to its own pair of files, so no lock is
                # needed for the writes themselves, only for the shared pbar
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                written += 1
                with pbar_lock:
                    pbar.update(1)
    except Exception as e:
        return f"{warc_filename}: failed to open ({e})"

    return f"{warc_filename}: {written} written, {skipped} skipped (existing), {failed} failed"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    grouped, id_width = load_records(RECORDS_GLOB)
    total_files = len(grouped)
    total_records = sum(len(v) for v in grouped.values())
    print(f"{total_records} records across {total_files} WARC files "
          f"(id width: {id_width} digits). Running with {MAX_WORKERS} concurrent workers.")

    with tqdm(total=total_records, unit="record", desc="Extracting") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_warc_file, warc_filename, entries, pbar): warc_filename
                for warc_filename, entries in grouped.items()
            }
            for future in as_completed(futures):
                tqdm.write(future.result())


if __name__ == "__main__":
    main()
