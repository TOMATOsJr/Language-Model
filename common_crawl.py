import gzip
import io
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator


# ============================================================
# CONFIG
# ============================================================

CRAWL = "CC-MAIN-2026-25"

DOMAIN = "onlinekhabar.com"

# MAX_PAGES = 100

OUTPUT_DIR = Path("nepali_corpus")

HEADERS = {
    "User-Agent": "NepaliCorpusResearch/0.1"
}

INDEX_URL = (
    f"https://index.commoncrawl.org/"
    f"{CRAWL}-index"
)

DATA_URL = "https://data.commoncrawl.org/"

# ============================================================
# CHECK FOR DUPLICATE URL
# ============================================================
def deduplicate_records(records):
    """
    Keep only one Common Crawl record per article URL.

    If the same URL appears multiple times in the index,
    keep the first record encountered.
    """

    unique = {}

    for record in records:

        url = record.get("url")

        if not url:
            continue

        # Remove URL fragments
        url = url.split("#", 1)[0]

        if url not in unique:
            unique[url] = record

    return list(unique.values())

# ============================================================
# QUERY COMMON CRAWL
# ============================================================

# def query_commoncrawl(domain, limit=100): // uses the limit parameter
def query_commoncrawl(domain):

    params = {
        "url": f"{domain}/*",
        "output": "json",
        "filter": "status:200",
        # "limit": limit,
    }

    response = requests.get(
        INDEX_URL,
        params=params,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    records = []

    for line in response.text.splitlines():

        if not line.strip():
            continue

        records.append(json.loads(line))

    return records


# ============================================================
# DOWNLOAD WARC RECORD
# ============================================================

def download_warc_record(record):

    filename = record["filename"]
    offset = int(record["offset"])
    length = int(record["length"])

    start = offset
    end = offset + length - 1

    url = DATA_URL + filename

    headers = {
        **HEADERS,
        "Range": f"bytes={start}-{end}",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=120,
    )

    response.raise_for_status()

    return response.content


# ============================================================
# EXTRACT HTML FROM WARC
# ============================================================

def extract_html(warc_bytes):

    # The individual WARC record is gzip compressed.
    decompressed = gzip.decompress(warc_bytes)

    stream = io.BytesIO(decompressed)

    for record in ArchiveIterator(stream):

        if record.rec_type != "response":
            continue

        payload = record.content_stream().read()

        return payload

    return None


# ============================================================
# HTML -> TEXT
# ============================================================

def html_to_text(html_bytes):

    soup = BeautifulSoup(
        html_bytes,
        "html.parser",
    )

    # Remove things we definitely don't want
    # in the initial extracted representation.
    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
    ]):
        tag.decompose()

    text = soup.get_text(
        separator="\n",
    )

    # Basic whitespace normalization.
    # This is NOT our actual cleaning pipeline.
    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def extract_category_from_url(url):

    if not url:
        return "uncategorized"

    path = urlparse(url).path

    segments = [
        segment.strip()
        for segment in path.split("/")
        if segment.strip()
    ]

    # Setopati article URLs generally start with /<category>/...
    if not segments:
        return "uncategorized"

    category = segments[0].lower()

    # Keep directory names filesystem-safe.
    category = re.sub(
        r"[^a-z0-9_-]+",
        "_",
        category,
    ).strip("_")

    return category or "uncategorized"


# ============================================================
# SAVE
# ============================================================

def save_document(index, record, html, text):

    category = extract_category_from_url(
        record.get("url")
    )

    category_dir = OUTPUT_DIR / category

    category_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    html_path = (
        category_dir /
        f"{index:06d}.html"
    )

    text_path = (
        category_dir /
        f"{index:06d}.txt"
    )

    metadata_path = (
        category_dir /
        f"{index:06d}.json"
    )

    html_path.write_bytes(html)

    text_path.write_text(
        text,
        encoding="utf-8",
    )

    metadata = {
        "url": record.get("url"),
        "timestamp": record.get("timestamp"),
        "status": record.get("status"),
        "mime": record.get("mime"),
        "languages": record.get("languages"),
        "encoding": record.get("encoding"),
        "digest": record.get("digest"),
        "filename": record.get("filename"),
        "offset": record.get("offset"),
        "length": record.get("length"),
        "crawl": CRAWL,
        "category": category,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        f"Searching Common Crawl: {DOMAIN}"
    )

    # records = query_commoncrawl(
    #     DOMAIN,
    #     MAX_PAGES,
    # )

    records = query_commoncrawl(
        DOMAIN,
    )

    print(
        f"Found {len(records)} records"
    )

    records = deduplicate_records(records)

    print(
        f"Found {len(records)} unique URLs"
    )

    saved = 0

    for record in tqdm(records):

        try:

            print(
                "\n",
                record["url"],
            )

            # Useful sanity check
            print(
                "Language:",
                record.get("languages"),
            )

            warc = download_warc_record(
                record
            )

            html = extract_html(
                warc
            )

            if html is None:
                print(
                    "No HTML response found"
                )
                continue

            text = html_to_text(
                html
            )

            if not text.strip():
                print(
                    "Empty text"
                )
                continue

            save_document(
                saved,
                record,
                html,
                text,
            )

            saved += 1

            print(
                f"Saved document {saved}"
            )

        except Exception as e:

            print(
                "ERROR:",
                repr(e),
            )

        time.sleep(0.2)

    print()
    print(
        f"Finished. Saved {saved} documents."
    )


if __name__ == "__main__":
    main()
