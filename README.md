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

## Cleaning and Pre-processing
1. Remove the extremely short docs. (like less than 100 words may be)
1. Remove Garbage docs (like 404 error or loading etc.,)
1. Remove repetitive docs (may be using hashing)
1. Unicode normalization (something like NFC)
1. Language identification using cld2 or 3 but may not necessary as common crawl already runs cld2
1. Score devanagri script ratio in the docs. (threshold like > 80 or > 90)
1. Remove gibberish as well
1. Character level quality filtering (give quality scoring based on these)
   - alphabetic character ratio
   - digits ratio
   - punctuation ratio
   - weird unicode ratio
   - repeated characters ratio
1. Normalize whitespaces (keep paragraph seperation or not ?)
1. Don't overclean punctuation and also normalize quotes
1. Normalize the english digits and devanagri ones (is there a reason to do or not do this ?)
1. Handle things like the Purna Viram (|) and Full Stop (.).
1. Replace URLs with \<URL\> may be for emails as well
1. Deduplication (remember to include the manual one in case of exact deduplication of records or something)
    - Paragraph dedup
    - Near duplicate removal using MinHash (sentence level as well)
    - Exact dedup
1. Perplexity based filtering (optional)
1. Once the document are clean, splitting them into sentences (may or may not be needed)
1. Safety filtering on the content

### Hindi runs
python3 ../cleaning/pipeline.py --input combined_hindi_corpus_copy.jsonl --output cleaned.jsonl
Processing records: 398288rec [30:31, 217.46rec/s]
Wrote 300530 cleaned records to cleaned.jsonl
Removal stats:
  - exact_duplicate: 30910
  - garbage_document: 13
  - gibberish: 3915
  - low_devanagari_ratio: 379
  - low_quality_characters: 2476
  - non_hindi: 5414
  - short_document: 54651
Total processed: 398288
Malformed JSON: 0
Empty records: 0
Word Count = 309049201

run-2 with near dedup
Processing records: 398288rec [1:12:41, 91.32rec/s]
Wrote 280245 cleaned records to cleaned2.jsonl
Removal stats:
  - exact_duplicate: 30910
  - garbage_document: 13
  - gibberish: 888
  - low_devanagari_ratio: 23
  - low_quality_characters: 2415
  - near_duplicate: 14155
  - non_hindi: 14988
  - short_document: 54651
Total processed: 398288
Malformed JSON: 0
Empty records: 0
Word Count = 289104595

In the cleaned2.jsonl
Documents:              280,245
Alphabetic characters:   759,047,664
Devanagari characters:   732,600,227
Devanagari percentage:   96.52%


### Nepali runs
With Deduplication
python3 ../cleaning/pipeline.py --input combined_nepali_corpus_copy.jsonl --output cleaned2.jsonl
Processing records: 316092rec [34:36, 152.25rec/s]
Wrote 236748 cleaned records to cleaned2.jsonl
Removal stats:
  - exact_duplicate: 12229
  - gibberish: 187
  - low_quality_characters: 88
  - near_duplicate: 15569
  - non_nepali: 8552
  - short_document: 42719
Total processed: 316092
Malformed JSON: 0
Empty records: 0
Word Count = 106803363

In the cleaned2.jsonl
Documents:              236,748
Alphabetic characters:   368,021,652
Devanagari characters:   365,471,134
Devanagari percentage:   99.31%

In the cleaned_ppl.jsonl
Word Count = 92365343