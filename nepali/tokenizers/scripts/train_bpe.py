#!/usr/bin/env python3

import argparse
import json
import re

import sentencepiece as spm

# Sentence boundary characters relevant to Devanagari-script text (danda,
# double danda) plus standard latin punctuation that shows up in mixed web
# text. This is the same boundary set used elsewhere in the cleaning
# pipeline, kept consistent here.
SENTENCE_END_RE = re.compile(r"(?<=[\u0964\u0965.!?])\s+")

def split_into_sentences(text: str):
    """
    Split a document into sentence-like units.

    Boundaries:
        ।  Devanagari danda
        ॥  Devanagari double danda
        .  Latin period
        !  Exclamation mark
        ?  Question mark

    Punctuation remains attached to the preceding sentence.

    No additional chunking or reconstruction is performed.
    """

    for sentence in SENTENCE_END_RE.split(text):

        sentence = sentence.strip()

        if sentence:
            yield sentence

def jsonl_sentence_iterator(
    file_path: str,
    text_field: str = "text",
):
    """
    Stream text from JSONL.

    For each JSONL record:

        JSON record
             ↓
        extract text field
             ↓
        sentence splitting using SENTENCE_END_RE
             ↓
        yield each sentence immediately

    The entire dataset is never loaded into memory.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue
            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            text = record.get(
                text_field,
                "",
            )

            if not isinstance(
                text,
                str,
            ):
                continue

            text = text.strip()

            if not text:
                continue

            # ------------------------------------------------
            # Split this record into sentences and immediately
            # yield them to SentencePiece.
            # ------------------------------------------------

            for sentence in split_into_sentences(text):

                yield sentence

def train(args):
    paths = args.input
    print(f"Training SentencePiece unigram model from: {', '.join(paths)}")

    iterator = jsonl_sentence_iterator(paths, text_field=args.text_field)

    spm.SentencePieceTrainer.train(
        sentence_iterator=iterator,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=args.character_coverage,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,   # reservoir-samples from the stream when input_sentence_size caps it
        byte_fallback=True,            # graceful handling of any char not in the learned vocab, rather than <unk>
        # normalization_rule_name="identity",  # corpus already went through our own NFC/whitespace cleaning --
                                              # avoid SentencePiece's default nmt_nfkc re-normalizing on top of it
        num_threads=args.num_threads,
        max_sentence_length=args.max_sentence_chars + 100,  # small headroom above our own pre-chunk limit
        split_digits=True,  # SentencePiece's default is False; we want to split digits into subwords for better generalization
        max_sentencepiece_length=32,  # SentencePiece default is 32; we keep it the same
        user_defined_symbols=["<URL>", "<EMAIL>", '।', '?', '!', ',', ':', ';', '-', '(', ')', '"', "'"],
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    print(f"\nDone. Model written to {args.model_prefix}.model / {args.model_prefix}.vocab")


def main():
    ap = argparse.ArgumentParser(description="Train a SentencePiece unigram tokenizer from JSONL corpora.")
    ap.add_argument("--input", required=True, help="One or more JSONL files.")
    ap.add_argument("--text-field", default="text", help="JSON field containing the document text. Default: text")
    ap.add_argument("--model-prefix", default="tokenizer", help="Output prefix for .model/.vocab. Default: tokenizer")
    ap.add_argument("--vocab-size", type=int, default=8000, help="Vocabulary size. Default: 8000")
    ap.add_argument(
        "--character-coverage", type=float, default=0.999,
        help="Fraction of characters to cover. Devanagari has many conjunct forms, so this is set higher "
             "than SentencePiece's 0.9995 default. Default: 0.999",
    )
    ap.add_argument(
        "--max-sentence-chars", type=int, default=50000,
        help="Hard cap (characters) on any single line fed to SentencePiece, after sentence splitting. Default: 1000",
    )
    ap.add_argument(
        "--input-sentence-size", type=int, default=1_000_000,
        help="Max sentences SentencePiece samples from the stream (reservoir sampling). Default: 10,000,000. "
             "Set to 0 to use every sentence in the corpus (slower, much more memory).",
    )
    ap.add_argument("--num-threads", type=int, default=8, help="SentencePiece training threads. Default: 8")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()