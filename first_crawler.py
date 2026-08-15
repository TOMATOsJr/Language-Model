import asyncio
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
import aiosqlite
import trafilatura
from bs4 import BeautifulSoup


BASE_URL = "https://www.setopati.com"

# Start small. Add more once you verify the output.
CATEGORIES = [
    "politics",
    "society",
    "business",
    "opinion",
    "blog",
    "literature",
    "sports",
    "global",
    "cover-story",
]

OUTPUT_DIR = Path("setopati_data")
DB_PATH = OUTPUT_DIR / "setopati.db"
OUTPUT_FILE = OUTPUT_DIR / "articles.jsonl"

# IMPORTANT:
# Keep this relatively conservative initially.
CONCURRENT_REQUESTS = 5
REQUEST_DELAY = 0.5

# Number of category pages to inspect.
# Set to None to continue until pages stop producing articles.
# MAX_PAGES_PER_CATEGORY = 100

USER_AGENT = (
    "NepaliCorpusResearch/1.0 "
    "(academic language-corpus project)"
)


# ---------------------------------------------------------
# Database
# ---------------------------------------------------------

async def init_db():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS urls (
                url TEXT PRIMARY KEY,
                category TEXT,
                status TEXT DEFAULT 'discovered',
                attempts INTEGER DEFAULT 0,
                http_status INTEGER,
                discovered_at REAL,
                downloaded_at REAL,
                error TEXT
            )
        """)

        await db.commit()


async def add_url(db, url, category):
    await db.execute("""
        INSERT OR IGNORE INTO urls
        (url, category, status, discovered_at)
        VALUES (?, ?, 'discovered', ?)
    """, (url, category, time.time()))


async def get_pending_urls(db, limit=100):
    cursor = await db.execute("""
        SELECT url, category
        FROM urls
        WHERE status = 'discovered'
        ORDER BY discovered_at
        LIMIT ?
    """, (limit,))

    return await cursor.fetchall()


# ---------------------------------------------------------
# Robots.txt
# ---------------------------------------------------------

def load_robots():
    robots_url = urljoin(BASE_URL, "/robots.txt")

    rp = RobotFileParser()
    rp.set_url(robots_url)

    try:
        rp.read()
        print(f"[robots] Loaded {robots_url}")
        return rp
    except Exception as e:
        print(f"[robots] Could not read robots.txt: {e}")
        return None


# ---------------------------------------------------------
# HTTP
# ---------------------------------------------------------

class RateLimiter:

    def __init__(self, delay):
        self.delay = delay
        self.lock = asyncio.Lock()
        self.last_request = 0

    async def wait(self):
        async with self.lock:
            now = time.monotonic()

            wait_time = self.delay - (now - self.last_request)

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            self.last_request = time.monotonic()


async def fetch(session, url, limiter, robots):
    if robots is not None:
        if not robots.can_fetch(USER_AGENT, url):
            print(f"[robots] blocked: {url}")
            return None, None

    await limiter.wait()

    for attempt in range(3):
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=30),
                allow_redirects=True,
            ) as response:

                status = response.status

                if status == 200:
                    text = await response.text(
                        errors="replace"
                    )
                    return text, status

                if status in {429, 500, 502, 503, 504}:
                    await asyncio.sleep(2 ** attempt)
                    continue

                return None, status

        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
        ) as e:

            if attempt == 2:
                print(f"[error] {url}: {e}")
                return None, None

            await asyncio.sleep(2 ** attempt)

    return None, None


# ---------------------------------------------------------
# Article URL discovery
# ---------------------------------------------------------

ARTICLE_PATTERN = re.compile(
    r"^https?://(?:www\.)?setopati\.com/"
    r"[^/?#]+/\d+/?$"
)


def is_article_url(url):
    parsed = urlparse(url)

    if parsed.netloc not in {
        "www.setopati.com",
        "setopati.com",
    }:
        return False

    path = parsed.path.rstrip("/")

    return bool(
        re.match(r"^/[^/]+/\d+$", path)
    )


def extract_article_urls(html):
    soup = BeautifulSoup(html, "html.parser")

    urls = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        url = urljoin(BASE_URL, href)

        # Remove fragments
        url = url.split("#")[0]

        if is_article_url(url):
            urls.add(url)

    return urls


# ---------------------------------------------------------
# Category crawler
# ---------------------------------------------------------

async def crawl_category(
    session,
    db,
    limiter,
    robots,
    category,
):
    print(f"\n=== CATEGORY: {category} ===")

    page = 1

    while True:

        # if (
        #     MAX_PAGES_PER_CATEGORY is not None
        #     and page > MAX_PAGES_PER_CATEGORY
        # ):
        #     break

        if page == 1:
            url = f"{BASE_URL}/{category}"
        else:
            url = f"{BASE_URL}/{category}?page={page}"

        print(f"[category] {url}")

        html, status = await fetch(
            session,
            url,
            limiter,
            robots,
        )

        if html is None:
            print(
                f"[category] failed page {page}, "
                f"status={status}"
            )
            break

        article_urls = extract_article_urls(html)

        print(
            f"[category] page={page} "
            f"articles={len(article_urls)}"
        )

        if not article_urls:
            print(
                f"[category] no articles on page {page}; "
                f"stopping"
            )
            break

        for article_url in article_urls:
            await add_url(
                db,
                article_url,
                category,
            )

        await db.commit()

        page += 1


# ---------------------------------------------------------
# Article extraction
# ---------------------------------------------------------

def extract_article(html, url):
    """
    Try JSON-LD first, then Trafilatura.
    """

    soup = BeautifulSoup(html, "html.parser")

    title = None
    date_published = None
    author = None

    # -----------------------------------------------------
    # JSON-LD
    # -----------------------------------------------------

    scripts = soup.find_all(
        "script",
        type="application/ld+json",
    )

    for script in scripts:

        if not script.string:
            continue

        try:
            data = json.loads(script.string)
        except Exception:
            continue

        candidates = []

        if isinstance(data, dict):
            candidates.append(data)

            # Some pages use @graph
            if isinstance(data.get("@graph"), list):
                candidates.extend(data["@graph"])

        elif isinstance(data, list):
            candidates.extend(data)

        for item in candidates:

            if not isinstance(item, dict):
                continue

            item_type = item.get("@type")

            if item_type in {
                "NewsArticle",
                "Article",
                "BlogPosting",
            }:

                title = item.get("headline")
                date_published = item.get(
                    "datePublished"
                )

                author_data = item.get("author")

                if isinstance(author_data, dict):
                    author = author_data.get("name")

                elif isinstance(author_data, list):
                    names = []

                    for a in author_data:
                        if isinstance(a, dict):
                            if a.get("name"):
                                names.append(a["name"])

                    author = ", ".join(names)

                # articleBody is extremely useful
                article_body = item.get("articleBody")

                if article_body:
                    return {
                        "url": url,
                        "title": title,
                        "author": author,
                        "date_published": date_published,
                        "text": article_body,
                        "extractor": "jsonld",
                    }

    # -----------------------------------------------------
    # Trafilatura fallback
    # -----------------------------------------------------

    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        include_links=False,
        favor_precision=True,
    )

    if not text:
        return None

    if title is None:
        h1 = soup.find("h1")

        if h1:
            title = h1.get_text(
                " ",
                strip=True,
            )

    return {
        "url": url,
        "title": title,
        "author": author,
        "date_published": date_published,
        "text": text,
        "extractor": "trafilatura",
    }


# ---------------------------------------------------------
# Basic quality filtering
# ---------------------------------------------------------

DEVANAGARI_RE = re.compile(
    r"[\u0900-\u097F]"
)


def devanagari_ratio(text):
    if not text:
        return 0

    devanagari = len(
        DEVANAGARI_RE.findall(text)
    )

    return devanagari / len(text)


def is_good_article(article):
    if not article:
        return False

    text = article["text"]

    # Don't keep tiny pages
    if len(text) < 500:
        return False

    # Require a reasonable amount of Devanagari
    if devanagari_ratio(text) < 0.25:
        return False

    return True


# ---------------------------------------------------------
# Download worker
# ---------------------------------------------------------

async def process_article(
    session,
    db,
    limiter,
    robots,
    url,
    category,
):

    html, status = await fetch(
        session,
        url,
        limiter,
        robots,
    )

    if html is None:

        await db.execute("""
            UPDATE urls
            SET status = 'failed',
                attempts = attempts + 1,
                http_status = ?,
                error = ?
            WHERE url = ?
        """, (
            status,
            f"HTTP status: {status}",
            url,
        ))

        await db.commit()

        return

    article = extract_article(
        html,
        url,
    )

    if not is_good_article(article):

        await db.execute("""
            UPDATE urls
            SET status = 'rejected',
                attempts = attempts + 1,
                http_status = ?,
                downloaded_at = ?
            WHERE url = ?
        """, (
            status,
            time.time(),
            url,
        ))

        await db.commit()

        return

    article["source"] = "setopati"
    article["category_discovered"] = category
    article["language"] = "ne"

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                article,
                ensure_ascii=False,
            )
            + "\n"
        )

    await db.execute("""
        UPDATE urls
        SET status = 'extracted',
            attempts = attempts + 1,
            http_status = ?,
            downloaded_at = ?
        WHERE url = ?
    """, (
        status,
        time.time(),
        url,
    ))

    await db.commit()

    print(
        f"[OK] {url} "
        f"chars={len(article['text'])}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

async def main():

    await init_db()

    robots = load_robots()

    limiter = RateLimiter(
        REQUEST_DELAY
    )

    connector = aiohttp.TCPConnector(
        limit=CONCURRENT_REQUESTS
    )

    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "ne-NP,ne;q=0.9,en;q=0.5",
    }

    async with aiohttp.ClientSession(
        connector=connector,
        headers=headers,
    ) as session:

        # -------------------------------------------------
        # STEP 1:
        # Discover article URLs
        # -------------------------------------------------

        async with aiosqlite.connect(
            DB_PATH
        ) as db:

            for category in CATEGORIES:

                await crawl_category(
                    session,
                    db,
                    limiter,
                    robots,
                    category,
                )

        # -------------------------------------------------
        # STEP 2:
        # Download discovered URLs
        # -------------------------------------------------

        async with aiosqlite.connect(
            DB_PATH
        ) as db:

            while True:

                rows = await get_pending_urls(
                    db,
                    limit=CONCURRENT_REQUESTS,
                )

                if not rows:
                    break

                tasks = [
                    process_article(
                        session,
                        db,
                        limiter,
                        robots,
                        url,
                        category,
                    )
                    for url, category in rows
                ]

                await asyncio.gather(*tasks)

                print(
                    f"[queue] processed {len(tasks)}"
                )


if __name__ == "__main__":
    asyncio.run(main())