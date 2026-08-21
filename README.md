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
With Dedup
python3 ../cleaning/pipeline.py --input combined_hindi_corpus_copy.jsonl --output cleaned.jsonl
Processing records: 398288rec [1:19:43, 83.25rec/s]
Wrote 280245 cleaned records to cleaned.jsonl
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
Word Count = 289074480

Perplexity Filtering
Percentiles:
P1     = 97.8483
P5     = 183.8402
P10    = 262.6902
P25    = 478.6698
P50    = 1029.2511
P75    = 2818.4101
P90    = 10538.5794
P95    = 34610.1930
P99    = 308546.6750
P99.5  = 404011.6576
P99.9  = 733961.145
============================================================
FILTER CONFIGURATION
============================================================
Bottom percentile filtering: disabled
Top 25.0%: PPL >= 2818.4101
Maximum PPL threshold: disabled
Minimum PPL threshold: 5.0000

============================================================
FILTER RESULTS
============================================================
Input scored      : 280,245
Removed bottom    : 0
Removed top       : 70,062
Removed threshold : 10
Total removed     : 70,072
Kept              : 210,173
Output             : cleaned_ppl.jsonl
Statistics: perplexity_stats.json
Word Count = 247580497

Documents:              280,245
Alphabetic characters:   759,047,664
Devanagari characters:   732,600,227
Devanagari percentage:   96.52%

============================================================
Hindi BLOCKLIST FILTER
============================================================
Input records       : 210,173
Invalid/skipped     : 0
Blocked/removed     : 356
Kept                : 209,817
Removed percentage  : 0.17%
Blocklist words     : 134
Maximum blocked ratio: 2.00%
Output              : manual_cleaned.jsonl
Word Count = 247138932

Manual documents kept : 81,550
Manual words kept     : 99,998,974

Target total words : 500,000,000
Manual words        : 99,998,974
Sangraha needed     : 400,001,026
Deduplicating Sangraha: 932276rec [21:25, 1043.10rec/s]
Reached target word count.
Deduplicating Sangraha: 932387rec [21:26, 725.01rec/s]

======================================================================
FINAL DATASET
======================================================================
Manual records        : 81,550
Manual words          : 99,998,974
Sangraha processed    : 932,388
Sangraha kept         : 932,270
Sangraha duplicates   : 64
Sangraha words kept   : 400,007,142
FINAL WORDS           : 500,006,116
TARGET WORDS          : 500,000,000
Output                : final_dataset.jsonl

Train Test Val Split
Reading final_dataset.jsonl...
Processed 1,000,000 records | 493,039,617 words

======================================================================
CORPUS
======================================================================
Total words:     498,874,342
Manual words:    99,438,514
Sangraha words:  399,435,828
Manual ratio:    19.933%
Sangraha ratio:  80.067%

Shuffling manual corpus...
Shuffling Sangraha corpus...

======================================================================
TARGET SPLIT
======================================================================
Train:       357,142,857 words (~500,000,000 tokens)
Validation:  70,865,742 words
Test:        70,865,742 words

Writing datasets...

======================================================================
FINAL DATASET
======================================================================

TRAIN
  Total words:     357,154,414
  Estimated tokens:500,016,180
  Manual:          71,193,055
  Sangraha:        285,961,359
  Manual ratio:    19.933%

VALIDATION
  Total words:     70,867,645
  Estimated tokens:99,214,703
  Manual:          14,126,837
  Sangraha:        56,740,808
  Manual ratio:    19.934%

TEST
  Total words:     70,852,283
  Estimated tokens:99,193,196
  Manual:          14,118,622
  Sangraha:        56,733,661
  Manual ratio:    19.927%

======================================================================
FILES
======================================================================
splits/train.jsonl
splits/validation.jsonl
splits/test.jsonl

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

Perplexity Filtering
In the cleaned_ppl.jsonl
Percentiles:
P1     = 37.8077
P5     = 236.1216
P10    = 382.8141
P25    = 705.2136
P50    = 1356.1301
P75    = 2903.0493
P90    = 7281.0680
P95    = 13811.1519
P99    = 63039.7616
P99.5  = 107390.6922
P99.9  = 434021.6703

============================================================
FILTER CONFIGURATION
============================================================
Bottom percentile filtering: disabled
Top 10.0%: PPL >= 7281.0680
Maximum PPL threshold: disabled
Minimum PPL threshold: 5.0000

============================================================
FILTER RESULTS
============================================================
Input scored      : 236,748
Removed bottom    : 0
Removed top       : 23,675
Removed threshold : 5
Total removed     : 23,680
Kept              : 213,068
Output             : cleaned_ppl.jsonl
Statistics: perplexity_stats.json
Word Count = 92365343

After Safety filtering
============================================================
NEPALI BLOCKLIST FILTER
============================================================
Input records       : 213,068
Invalid/skipped     : 0
Blocked/removed     : 1,239
Kept                : 211,829
Removed percentage  : 0.58%
Blocklist words     : 98
Maximum blocked ratio: 2.00%
Output              : manual_cleaned.jsonl
Word Count = 92448649

Final Merging
Manual words kept     : 93,325,568

Target total words : 500,000,000
Manual words        : 93,325,568
Sangraha needed     : 406,674,432
Deduplicating Sangraha: 1353187rec [23:30, 795.94rec/s]
Reached target word count.
Deduplicating Sangraha: 1353273rec [23:30, 959.11rec/s]

======================================================================
FINAL DATASET
======================================================================
Manual records        : 211,829
Manual words          : 93,325,568
Sangraha processed    : 1,353,274
Sangraha kept         : 1,352,548
Sangraha duplicates   : 726
Sangraha words kept   : 406,674,795
FINAL WORDS           : 500,000,363
TARGET WORDS          : 500,000,000
Output                : final_dataset.jsonl

Train Test Val Split
======================================================================
CORPUS
======================================================================
Total words:     498,153,961
Manual words:    92,916,008
Sangraha words:  405,237,953
Manual ratio:    18.652%
Sangraha ratio:  81.348%

Shuffling manual corpus...
Shuffling Sangraha corpus...

======================================================================
TARGET SPLIT
======================================================================
Train:       357,142,857 words (~500,000,000 tokens)
Validation:  70,505,552 words
Test:        70,505,552 words

Writing datasets...

======================================================================
FINAL DATASET
======================================================================

TRAIN
  Total words:     357,143,405
  Estimated tokens:500,000,767
  Manual:          66,614,806
  Sangraha:        290,528,599
  Manual ratio:    18.652%

VALIDATION
  Total words:     70,507,150
  Estimated tokens:98,710,010
  Manual:          13,151,850
  Sangraha:        57,355,300
  Manual ratio:    18.653%

TEST
  Total words:     70,503,406
  Estimated tokens:98,704,768
  Manual:          13,149,352
  Sangraha:        57,354,054
  Manual ratio:    18.651%

======================================================================
FILES
======================================================================
splits/train.jsonl
splits/validation.jsonl
splits/test.jsonl