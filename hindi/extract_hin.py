import csv
import io
import json
import logging
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import trafilatura
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator

# Log failures clearly
logging.basicConfig(
    filename="download_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(message)s",
)

INDEX_CSV = Path("hindi_only.csv")
OUTPUT_DIR = Path("hindi_corpus")

# Official Hugging Face Common Crawl Storage Bucket HTTPS endpoint
DATA_URL = "https://huggingface.co/buckets/commoncrawl/commoncrawl/resolve/"

# Replace with your free Hugging Face token (starts with hf_...) to ensure smooth high-speed access
HF_TOKEN = ""

# 30 workers provides safe, high-speed execution
MAX_WORKERS = 30

# Persistent session with authentication headers
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "HindiCorpusResearch/0.5 (Research Project)",
    "Authorization": f"Bearer {HF_TOKEN}"
})


def download_and_extract_bytes(filename, offset, length):
    start = offset
    end = offset + length - 1

    url = DATA_URL + filename
    headers = {"Range": f"bytes={start}-{end}"}

    max_retries = 5
    for attempt in range(max_retries):
        try:
            resp = SESSION.get(url, headers=headers, timeout=25)

            if resp.status_code == 206: # Partial Content (Success)
                stream = io.BytesIO(resp.content)
                for record in ArchiveIterator(stream):
                    if record.rec_type == "response":
                        return record.content_stream().read()
                return None

            elif resp.status_code in [403, 429, 503]:
                wait_time = (2 ** attempt) + random.uniform(1, 3)
                logging.warning(f"Throttled (Status {resp.status_code}) on {filename}. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
                continue

            else:
                resp.raise_for_status()

        except (requests.exceptions.RequestException, ConnectionResetError) as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to download {url} after {max_retries} attempts. Error: {e}")
                return None
            time.sleep((2 ** attempt) + random.uniform(1, 3))

    return None


def extract_plain_text(html_bytes):
    try:
        return trafilatura.extract(
            html_bytes.decode("utf-8", errors="ignore"),
            include_comments=False,
            include_tables=True,
            no_fallback=False
        )
    except Exception:
        return None


def process_record(args):
    idx, row = args

    # Check if we already processed this record by verifying the text file exists
    txt_file = OUTPUT_DIR / f"{idx:08d}.txt"
    json_file = OUTPUT_DIR / f"{idx:08d}.json"
    if txt_file.exists() and json_file.exists():
        return "skipped"

    try:
        filename = row["warc_filename"]
        offset = int(row["warc_record_offset"])
        length = int(row["warc_record_length"])

        # Small stagger to prevent burst congestion
        time.sleep(random.uniform(0.05, 0.2))

        body = download_and_extract_bytes(filename, offset, length)
        if not body:
            return "failed"

        # Extract plain text directly in memory (No HTML file written to disk)
        plain_text = extract_plain_text(body)

        if plain_text:
            txt_file.write_text(plain_text, encoding="utf-8")

        metadata = {
            "url": row.get("url"),
            "languages": row.get("content_languages"),
            "warc_filename": filename,
            "warc_offset": offset,
            "warc_length": length,
            "has_extracted_text": bool(plain_text)
        }
        json_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        return "success"
    except Exception as e:
        logging.error(f"Row {idx} failed ({row.get('url')}): {str(e)}")
        return "failed"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INDEX_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(enumerate(reader))

    # CRITICAL OPTIMIZATION: Sort by WARC file AND offset to hit mirror file byte-ranges sequentially
    print("Sorting records to optimize storage bucket byte-range pulls...")
    records.sort(key=lambda x: (x[1]['warc_filename'], int(x[1]['warc_record_offset'])))

    print(f"Loaded {len(records)} records. Downloading via Hugging Face Mirror with MAX_WORKERS={MAX_WORKERS}...")

    stats = {"success": 0, "skipped": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_record, record_tuple) for record_tuple in records]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            status = future.result()
            stats[status] += 1

    print("\n--- RESULTS SUMMARY ---")
    print(f"Successfully Saved: {stats['success']}")
    print(f"Skipped (Already Exists): {stats['skipped']}")
    print(f"Failed: {stats['failed']}")

if __name__ == "__main__":
    main()