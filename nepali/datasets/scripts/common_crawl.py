import gzip
import os
import urllib.request
import duckdb
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ---- tuning knobs ----
MAX_WORKERS = 12          # concurrent duckdb queries against Common Crawl. Start at 8-16;
                           # this is network/remote-scan bound, not CPU bound, so you can push
                           # higher, but too many parallel remote parquet scans can start
                           # tripping timeouts/throttling. Tune based on what you observe.
RETRIES = 3                # per-partition retry attempts on transient network errors
OUTPUT_DIR = "../nepali_chunks"

print_lock = Lock()


def get_partition_keys(paths_url: str) -> list[str]:
    req = urllib.request.Request(paths_url)
    with urllib.request.urlopen(req) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read().decode("utf-8")

    all_paths = [line.strip() for line in file_content.split("\n") if line.strip()]
    return [p for p in all_paths if "subset=warc/" in p]


def scan_partition(i: int, key: str, total: int) -> str:
    """Runs one duckdb query against one remote parquet partition and writes a CSV.
    Each call gets its own duckdb connection since connections aren't safe to
    share across threads."""
    chunk_file = os.path.join(OUTPUT_DIR, f"part_{i:04d}.csv")

    if os.path.exists(chunk_file):
        return f"[{i+1}/{total}] skipped (already exists)"

    exact_https_url = f"https://data.commoncrawl.org/{key}"

    query = f"""
    SELECT
        url,
        warc_filename,
        warc_record_offset,
        warc_record_length,
        content_languages
    FROM read_parquet('{exact_https_url}')
    WHERE content_languages LIKE '%nep%'
    """

    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            con = duckdb.connect()
            con.execute("LOAD httpfs;")
            con.execute(f"COPY ({query}) TO '{chunk_file}' (HEADER, DELIMITER ',')")
            con.close()
            return f"[{i+1}/{total}] done: {key.split('/')[-1]}"
        except Exception as e:
            last_err = e
            with print_lock:
                print(f"[{i+1}/{total}] attempt {attempt}/{RETRIES} failed: {e}")

    return f"[{i+1}/{total}] FAILED after {RETRIES} attempts: {last_err}"


def main():
    paths_url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-25/cc-index-table.paths.gz"

    # one-time install (persists on disk; workers below only need LOAD)
    setup_con = duckdb.connect()
    setup_con.execute("INSTALL httpfs;")
    setup_con.close()

    parquet_keys = get_partition_keys(paths_url)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = len(parquet_keys)
    print(f"Found {total} partitions. Scanning with {MAX_WORKERS} concurrent workers.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(scan_partition, i, key, total): i
            for i, key in enumerate(parquet_keys)
        }

        for future in as_completed(futures):
            with print_lock:
                print(future.result())


if __name__ == "__main__":
    main()