import requests
import json

CRAWL = "CC-MAIN-2026-25"

url = (
    f"https://index.commoncrawl.org/"
    f"{CRAWL}-index"
)

params = {
    "url": "onlinekhabar.com/*",
    "output": "json",
    "filter": "status:200",
    "limit": 10,
}

response = requests.get(
    url,
    params=params,
    headers={
        "User-Agent": "NepaliCorpusResearch/0.1"
    },
    timeout=60,
)

print("Status:", response.status_code)
print(response.url)
print()

for line in response.text.splitlines():
    print(json.loads(line))
