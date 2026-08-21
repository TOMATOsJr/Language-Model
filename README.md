# Hindi/Nepali Language Modeling Corpus

A data-processing and tokenizer-training pipeline for building a large-scale Hindi/Nepali corpus from manually curated and Sangraha/Common Crawl-derived text.

The pipeline covers:

1. Perplexity-based filtering
2. Hindi/Nepali blocklist filtering
3. Manual + Sangraha corpus merging
4. Deduplication and target-size selection (over manual + sangraha)
5. Train/validation/test splitting
6. SentencePiece Unigram tokenizer training
7. Tokenizer evaluation using fertility, unknown-token rate, token statistics, and tokenization examples

---

## 1. Pipeline Overview

```text
Raw Hindi/Nepali Web / Curated Data
            │
            ▼
    Initial Preprocessing
    (Normalization, Deduplication, etc,.)
            |
            ▼
   Perplexity Filtering
            │
            ▼
    Hindi/Nepali Blocklist Filter
            │
            ▼
       Cleaned Manual
            │
            │
            ├──────────────┐
            │              │
            ▼              ▼
       Manual Corpus    Sangraha Corpus
            │              │
            └──────┬───────┘
                   ▼
       Deduplication + Sampling
                   │
                   ▼
          ~500M Word Corpus
                   │
                   ▼
        Train / Validation / Test
                   │
                   ▼
       SentencePiece Unigram
                   │
                   ▼
      Tokenizer Evaluation
```

---

# 2. Corpus Construction

The final corpus combines two sources:

- **Manual**: manually curated/cleaned Hindi/Nepali data
- **Sangraha**: large-scale Hindi/Nepali web corpus data

The final target is approximately:

```text
500,000,000 words
```

with approximately:

```text
20% Manual
80% Sangraha
```

The split preserves this source ratio approximately across train, validation, and test.

So common crawl data was extracted for the manual part which was put through initial pre-processing of omitting bad records, normalization, exact dedup, near dedup, language filtering, etc.,

---

# 3. Perplexity Filtering

The first filtering stage uses language-model perplexity to identify documents that are likely to be noisy, low-quality, or anomalous.

## Perplexity Distribution

The observed perplexity percentiles were:

| Percentile | Perplexity |
|---:|---:|
| P1 | 97.85 |
| P5 | 183.84 |
| P10 | 262.69 |
| P25 | 478.67 |
| P50 | 1,029.25 |
| P75 | 2,818.41 |
| P90 | 10,538.58 |
| P95 | 34,610.19 |
| P99 | 308,546.68 |
| P99.5 | 404,011.66 |
| P99.9 | 733,961.15 |

The selected configuration removed the highest-perplexity 25% of documents, together with documents below the minimum perplexity threshold.

```text
---------------------------------------------------------------
FILTER CONFIGURATION
---------------------------------------------------------------
Bottom percentile filtering: disabled
Top 25.0%: PPL >= 2818.4101
Maximum PPL threshold: disabled
Minimum PPL threshold: 5.0000

---------------------------------------------------------------
FILTER RESULTS
---------------------------------------------------------------
Input scored      : 280,245
Removed bottom    : 0
Removed top       : 70,062
Removed threshold : 10
Total removed     : 70,072
Kept              : 210,173
Output            : cleaned_ppl.jsonl
Statistics        : perplexity_stats.json
Word Count        : 247,580,497

Documents:              280,245
Alphabetic characters:   759,047,664
Devanagari characters:   732,600,227
Devanagari percentage:    96.52%
```

---

# 4. Hindi/Nepali Blocklist Filtering

The perplexity-filtered corpus is then passed through a Hindi/Nepali-specific blocklist filter.

The filter removes documents whose blocked-word ratio exceeds the configured threshold.

Configuration/results:

```text
---------------------------------------------------------------
HINDI BLOCKLIST FILTER
---------------------------------------------------------------
Input records       : 210,173
Invalid/skipped     : 0
Blocked/removed     : 356
Kept                : 209,817
Removed percentage  : 0.17%
Blocklist words     : 134
Maximum blocked ratio: 2.00%
Output              : manual_cleaned.jsonl
Word Count          : 247,138,932

Manual documents kept : 81,550
Manual words kept     : 99,998,974
```

The resulting manual corpus contributes approximately 100M words to the final dataset.

---

# 5. Manual + Sangraha Corpus Construction

The final corpus has a target size of:

```text
500,000,000 words
```

The manually curated corpus contributes:

```text
99,998,974 words
```

Therefore the required Sangraha contribution is approximately:

```text
500,000,000
- 99,998,974
----------------
400,001,026 words
```

Sangraha records are processed, deduplicated, and retained until the required word target is reached.

```text
Target total words : 500,000,000
Manual words        : 99,998,974
Sangraha needed     : 400,001,026
```

Final corpus construction:

```text
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
```

The small amount above the target is a consequence of selecting whole records rather than cutting records in the middle.

---

# 6. Train / Validation / Test Split

The final dataset is shuffled separately by source and then split.

The corpus available to the split was:

```text
======================================================================
CORPUS
======================================================================
Total words:     498,874,342
Manual words:     99,438,514
Sangraha words:  399,435,828
Manual ratio:         19.933%
Sangraha ratio:       80.067%

Shuffling manual corpus...
Shuffling Sangraha corpus...
```

The target training size is chosen using the approximation:

```text
1 word ≈ 1.4 tokenizer tokens
```

Therefore:

```text
500,000,000 tokens / 1.4
≈ 357,142,857 words
```

The remaining words are divided approximately equally between validation and test.

## Target Split

```text
======================================================================
TARGET SPLIT
======================================================================
Train:       357,142,857 words (~500,000,000 tokens)
Validation:   70,865,742 words
Test:         70,865,742 words
```

## Actual Split

### Train

```text
TRAIN
  Total words:     357,154,414
  Estimated tokens:500,016,180
  Manual:           71,193,055
  Sangraha:        285,961,359
  Manual ratio:         19.933%
```

### Validation

```text
VALIDATION
  Total words:      70,867,645
  Estimated tokens: 99,214,703
  Manual:           14,126,837
  Sangraha:         56,740,808
  Manual ratio:         19.934%
```

### Test

```text
TEST
  Total words:      70,852,283
  Estimated tokens: 99,193,196
  Manual:           14,118,622
  Sangraha:         56,733,661
  Manual ratio:         19.927%
```

The Manual/Sangraha ratio therefore remains very close to the intended 20/80 distribution in all three splits.

---

# 7. Dataset Format

The datasets are stored as JSONL.

Each line represents one document/record and contains a `text` field:

```json
{"text": "हिंदी पाठ यहाँ है...", "source": "manual"}
```

or:

```json
{"text": "हिंदी पाठ यहाँ है...", "source": "sangraha"}
```

The tokenizer pipeline reads the JSONL directly rather than requiring an intermediate text file.

---

# 8. SentencePiece Unigram Tokenizer

The tokenizer uses **SentencePiece Unigram**.

The tokenizer is trained directly from the JSONL corpus using a streaming sentence iterator.

The processing flow is:

```text
JSONL record
     │
     ▼
record["text"]
     │
     ▼
Sentence boundary splitting
     │
     ▼
SentencePiece Unigram training
```

The dataset itself is not loaded entirely into memory.

## Sentence Splitting

Sentence boundaries are detected using:

```python
SENTENCE_END_RE = re.compile(
    r"(?<=[\u0964\u0965.!?])\s+"
)
```

This recognizes whitespace following:

- `।` — Devanagari danda
- `॥` — Devanagari double danda
- `.` — Latin period
- `!` — exclamation mark
- `?` — question mark

The punctuation remains attached to the preceding sentence.

For example:

```text
यह पहला वाक्य है। यह दूसरा वाक्य है? यह तीसरा वाक्य है!
```

becomes approximately:

```text
यह पहला वाक्य है।
यह दूसरा वाक्य है?
यह तीसरा वाक्य है!
```

No whitespace-based reconstruction or artificial chunking is performed.

---

# 9. SentencePiece Configuration

The tokenizer uses the following configuration:

```text
Model type:              Unigram
Character coverage:      0.9995
Input sentence sample:   1,000,000
Shuffle input:           enabled
Byte fallback:           enabled
Split digits:            enabled
Maximum piece length:    32
Training threads:        8
```

Special token IDs:

```text
<pad>  = 0
<unk>  = 1
<s>    = 2
</s>   = 3
```

User-defined symbols include:

```text
<URL>
<EMAIL>
।
?
!
,
:
;
-
(
)
"
'
```

`<URL>` and `<EMAIL>` are kept as user-defined symbols so that these placeholders can be represented as atomic tokenizer pieces.

---

# 10. Training Tokenizers at Different Vocabulary Sizes

Multiple vocabulary sizes can be trained to determine the best tokenizer configuration.

For example:

```bash
python3 train_bpe.py \
    --input ../datasets/splits/train.jsonl \
    --model-prefix models/unigram_8000 \
    --vocab-size 8000
```

For a 16K vocabulary:

```bash
python3 train_bpe.py \
    --input ../datasets/splits/train.jsonl \
    --model-prefix models/unigram_16000 \
    --vocab-size 16000
```

The resulting files are:

```text
models/
├── unigram_8000.model
├── unigram_8000.vocab
├── unigram_16000.model
├── unigram_16000.vocab
└── ...
```

---

# 11. Tokenizer Evaluation

Tokenizer quality is evaluated on the **held-out validation and test JSONL files**.

The evaluation reads the JSONL directly:

```text
validation.jsonl
       │
       ▼
JSON record
       │
       ▼
record["text"]
       │
       ▼
SentencePiece tokenizer
       │
       ├── fertility
       ├── UNK statistics
       ├── token frequencies
       ├── characters/token
       └── tokenization examples
```

No TXT intermediary is used.

---

# 12. Fertility

Fertility measures how many tokenizer tokens are generated per whitespace-separated word.

For each document:

```python
words = text.split()
token_ids = sp.encode(text, out_type=int)
```

The corpus-level metric is:

```text
Fertility = Total tokenizer tokens / Total whitespace words
```

A lower fertility generally means that the tokenizer represents each word with fewer pieces.

The vocabulary-size experiment therefore compares fertility across:

```text
8K
16K
24K
...
```

and uses the validation set to guide vocabulary-size selection.

---

# 13. Unknown Token Rate

The evaluation uses the actual `<unk>` ID stored in the SentencePiece model.

```text
UNK rate =
    UNK tokens / total tokenizer tokens
```

The evaluation also records:

- Total UNK tokens
- Number of records containing an UNK
- Percentage of records containing an UNK

Byte fallback is enabled during training, so unknown-character behavior can also be examined alongside the learned vocabulary.

---

# 14. Additional Tokenizer Statistics

The evaluation collects the following statistics.

## Token frequency

For every observed token:

```text
token
count
frequency
```

The most frequent tokens are also printed and saved.

## Average characters per token

Computed as:

```text
non-whitespace characters / total tokenizer tokens
```

This provides another measure of how much text is represented by each token.

## Tokenization examples

Example outputs include:

```text
Original text
Whitespace word count
SentencePiece pieces
Token IDs
```

For example:

```text
Text:
भारत एक विविधतापूर्ण देश है।

Pieces:
▁भारत | ▁एक | ▁विविधतापूर्ण | ▁देश | ▁है | ।

Token IDs:
...
```

---

# 15. Evaluation Outputs

The evaluation produces:

```text
evaluation/
├── tokenizer_results.csv
├── fertility_vs_vocab.png
├── unk_rate_vs_vocab.png
├── chars_per_token_vs_vocab.png
│
├── validation_unigram_8000_token_frequencies.csv
├── validation_unigram_8000_examples.txt
├── validation_unigram_16000_token_frequencies.csv
├── validation_unigram_16000_examples.txt
│
├── test_unigram_8000_token_frequencies.csv
├── test_unigram_8000_examples.txt
├── test_unigram_16000_token_frequencies.csv
├── test_unigram_16000_examples.txt
└── ...
```

Run evaluation with:

```bash
python3 evaluate_sentencepiece.py \
    --models models \
    --validation ../datasets/splits/validation.jsonl \
    --test ../datasets/splits/test.jsonl \
    --output-dir evaluation
```

---

# 16. Evaluation Metrics

The main metrics used for tokenizer comparison are:

| Metric | Description |
|---|---|
| Fertility | Tokens generated per whitespace word |
| UNK rate | Percentage of generated tokens that are `<unk>` |
| UNK record rate | Percentage of records containing at least one `<unk>` |
| Average characters/token | Non-whitespace characters represented per token |
| Unique observed tokens | Number of vocabulary pieces observed in held-out text |
| Token frequency | Distribution of token usage |
| Tokenization examples | Qualitative inspection of segmentation |

---

# 17. Vocabulary Size Selection

The tokenizer vocabulary size is treated as an experimental parameter.

The evaluation compares multiple vocabulary sizes using held-out validation data.

The primary trade-off is:

```text
Larger vocabulary
       │
       ├── Lower fertility
       ├── More characters per token
       └── Larger model vocabulary

Smaller vocabulary
       │
       ├── Higher fertility
       ├── Smaller vocabulary
       └── Potentially more granular segmentation
```

Unknown-token behavior is evaluated alongside fertility rather than selecting a vocabulary size from fertility alone.

The test set should be treated as a final held-out evaluation set rather than repeatedly using it for tokenizer selection.

---

# 18. Reproducibility

The complete corpus construction process should be reproducible from the intermediate datasets and configuration parameters.

Important parameters include:

```text
Perplexity filtering thresholds
Hindi/Nepali blocklist threshold
Manual/Sangraha target ratio
Target corpus word count
Train/validation/test proportions
Word-to-token estimate
Sentence boundary regex
SentencePiece model type
Vocabulary size
Character coverage
Input sentence sample size
Byte fallback
Digit splitting
User-defined symbols
```

---

# 19. Final Corpus Summary

The resulting corpus is approximately:

```text
~500M words
~20% Manual
~80% Sangraha
```

The training split is approximately:

```text
357M words
~500M estimated tokenizer tokens
```

with approximately:

```text
71M Manual words
286M Sangraha words
```

The validation and test sets each contain approximately:

```text
71M words
~99M estimated tokenizer tokens
```

---

# 20. Project Structure

A representative project layout is:

```text
hindi/
│
├── datasets/
│   ├── splits/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   │
│   └── ...
│
├── tokenizers/
│   ├── models/
│   │   ├── unigram_8000.model
│   │   ├── unigram_8000.vocab
│   │   ├── unigram_16000.model
│   │   └── unigram_16000.vocab
│   │
│   ├── evaluation/
│   │   ├── tokenizer_results.csv
│   │   ├── fertility_vs_vocab.png
│   │   ├── unk_rate_vs_vocab.png
│   │   └── chars_per_token_vs_vocab.png
│   │
│   └── scripts/
│       ├── train_bpe.py
│       └── evaluate_sentencepiece.py
│
└── README.md
```

---

# 21. End-to-End Summary

The complete pipeline is:

```text
                 RAW DATA
                    │
                    ▼
          Perplexity Scoring
                    │
                    ▼
          Perplexity Filtering
                    │
                    ▼
          Hindi/Nepali Blocklist Filter
                    │
                    ▼
             Manual Corpus
                    │
                    │
       ┌────────────┴────────────┐
       │                         │
       ▼                         ▼
    Manual                    Sangraha
       │                         │
       └────────────┬────────────┘
                    ▼
          Deduplication/Sampling
                    │
                    ▼
             ~500M words
                    │
                    ▼
          Source-balanced shuffle
                    │
                    ▼
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
      Train      Validation    Test
     ~357M        ~71M         ~71M
       │
       ▼
 Sentence Boundary Splitting
       │
       ▼
SentencePiece Unigram Training
       │
       ├──────────────┐
       ▼              ▼
    .model          .vocab
       │
       ▼
Held-out Tokenizer Evaluation
       │
       ├── Fertility
       ├── UNK rate
       ├── Chars/token
       ├── Token frequencies
       └── Tokenization examples
```

The resulting tokenizer and corpus can then be used as the data/tokenization foundation for subsequent Hindi/Nepali language-model training.
