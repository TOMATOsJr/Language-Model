## Dataset Accumulation
### Prebuilt corpus
1. Sangraha Nepali Corpus - Extracted 500M words
2. Sangraha Hindi Corpus - Extracted 500M words

### Manual collection
#### Consideration
1. NewsPaper Scraping
2. Government sites Scraping
3. Literature scraping (if possible and if necessary)
4. Open subtitles (seems to be a good idea)

#### Actually done
Data extraction from common crawl which includes all kind
of scraping of the web over various domains.

Performed extraction of the URLs using the duckdb database and
got the warc record offsets and later extracted the relevant
text out of these sites using trafilatura efficiently.