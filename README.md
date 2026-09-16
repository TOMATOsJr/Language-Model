## Phase - 1 submission link
[LMA_dataset](https://iiithydresearch-my.sharepoint.com/:f:/g/personal/jayanth_raju_research_iiit_ac_in/IgAX6U0juLSJTpCXe1W4Y8AfAbDCv1lgB6owA7AOWL0khfI?e=ns13lO)

## Phase - 2 One-Drive Pre-trained Language Model files link
[LMA_phase2](https://iiithydresearch-my.sharepoint.com/:f:/g/personal/jayanth_raju_research_iiit_ac_in/IgAeoRhtYHcaQLMnfDTLQyKoAS2VRVExQNLMWlCTbarPLPo?e=xo0Gm4)

## Phase - 3 One-Drive Finetuned models link + bonus
[LMA_phase3](https://iiithydresearch-my.sharepoint.com/:f:/g/personal/jayanth_raju_research_iiit_ac_in/IgB4Ep6fc1NJQrDJhsu3s0HxAVSIG3zT7c4nwP8CQ-ZIgcg?e=MebNZr)

# Hindi/Nepali Language Modeling Corpus

A data-processing and tokenizer-training pipeline for building a large-scale Hindi/Nepali corpus from manually curated and Sangraha/Common Crawl-derived text.

## Interactive Inference

Launch the Gradio interface from the repository root to generate text with either the Hindi or Nepali model:

```bash
source venv/bin/activate
pip install gradio
python gradio_app.py
```

The application uses `hindi/training/checkpoints/best.pt` and
`nepali/training/checkpoints/best.pt` by default and loads a model only when
it is selected. Open the local URL printed by Gradio, select a model, enter a
prompt, and adjust the sampling controls as needed.

To use different checkpoints or make the interface reachable on the network:

```bash
python gradio_app.py \
       --hindi-checkpoint /path/to/hindi.pt \
       --nepali-checkpoint /path/to/nepali.pt \
       --host 0.0.0.0 \
       --port 7860
```

## Pipeline Execution

The pipeline structure is identical for both Hindi and Nepali. Replace `<lang>` with either `hindi` or `nepali` where applicable.

---

### 1. Dataset (`<lang>/datasets/`)

Scripts to scrape and consolidate raw text from Common Crawl and other curated sources.

| Script | Description |
|---|---|
| `common_crawl.py` | Downloads raw Hindi/Nepali records from Common Crawl using DuckDB. |
| `extract_hin.py` / `extract_nep.py` | Extracts target-language text fields from the raw downloaded records. |
| `concate_dataset.py` | Concatenates multiple source JSONL files into a single corpus file. |
| `stats.py` | Prints token/document count statistics for the raw corpus. |

```bash
python <lang>/datasets/scripts/common_crawl.py
python <lang>/datasets/scripts/concate_dataset.py
```

---

### 2. Cleaning (`<lang>/cleaning/`)

An elaborate multi-stage cleaning pipeline that filters and normalises the raw corpus.

| Script | Description |
|---|---|
| `pipeline.py` | **Main entry point.** Runs all cleaning stages end-to-end on an input JSONL file. |
| `merge_corpora.py` | Merges multiple cleaned corpora into one before or after cleaning. |
| `various_normalization.py` | Unicode and script normalisation (Devanagari-specific). |
| `lang_doc_filtering.py` | Language-ID filtering using FastText to drop non-target-language docs. |
| `basic_doc_filtering.py` | Heuristic length and character-ratio filters. |
| `quality_filtering.py` | Perplexity-free quality signals (punctuation ratio, digit ratio, etc.). |
| `perplexity_filter.py` | KenLM-based perplexity filter to remove low-quality documents. |
| `sangraha_train_kenLM.py` | Extracts a clean text sample to train the KenLM language model. |
| `deduplication.py` | MinHash / exact-match deduplication across documents. |

```bash
# Run the full cleaning pipeline on a raw corpus
python <lang>/cleaning/pipeline.py --input raw_corpus.jsonl --output cleaned_corpus.jsonl
```

---

### 3. Tokenization (`<lang>/tokenizers/`)

Trains a SentencePiece tokenizer on the cleaned corpus.

```bash
python <lang>/tokenizers/scripts/train_bpe.py \
       --input cleaned_corpus.jsonl \
       --model-prefix <lang>/tokenizers/unigram_16000 \
       --vocab-size 16000
```

---

### 4. Pre-training (`<lang>/training/`)

Trains and evaluates the causal language model.

**Prepare binary data** — tokenises the cleaned corpus into a memory-mapped `.bin` file for fast loading:
```bash
python <lang>/training/prepare_data.py
```

**Train** — pre-trains the Transformer decoder from scratch:
```bash
python <lang>/training/train.py
```

**Generate** — autoregressive text generation from a trained checkpoint:
```bash
python <lang>/training/generate.py \
       --checkpoint <lang>/training/checkpoints/best.pt \
       --prompt "आज का मौसम"
```

**Intrinsic metrics** — reports cross-entropy, perplexity, and bits-per-byte:
```bash
python <lang>/training/intrinsic_metrics.py \
       --checkpoint <lang>/training/checkpoints/best.pt
```

**Attention analysis** — generates attention heatmaps and entropy/distance metrics:
```bash
python attention_analysis.py \
       --language <lang> \
       --checkpoint <lang>/training/checkpoints/best.pt \
       --prompt "आज का मौसम"
```

**Generation quality** — evaluates generation quality on held-out document prefixes (BLEU, repetition, entropy):
```bash
python generation_quality.py \
       --language <lang> \
       --checkpoint <lang>/training/checkpoints/best.pt
```

---

### 5. Synthetic Dataset Generation (`<lang>/generator/`)

Generates the structured synthetic reasoning dataset used for fine-tuning.

```bash
python <lang>/generator/dataset_pipeline.py \
       --train 20000 --val 2000 --test 2000 \
       --output-dir <lang>/generator/data
```

---

### 6. Fine-tuning (`<lang>/finetuning/`)

Fine-tunes the pre-trained model on the synthetic reasoning dataset and evaluates it.

**Train** — fine-tunes from the pre-trained checkpoint (auto-resumes from `latest.pt` if present):
```bash
python <lang>/finetuning/train.py
```

**Evaluate** — runs generation on the held-out test set and reports semantic accuracy metrics:
```bash
python <lang>/finetuning/evaluate_test.py \
       --checkpoint <lang>/finetuning/checkpoints/best.pt
```
