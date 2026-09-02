#!/usr/bin/env python3
"""
Tokenize the text data into token id sequences of size uint16.
While also adding special tokens for sentence boundaries and padding.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import sentencepiece as spm

# Language dir path so that we can easily locate the dataset
# tokenizer model files.

LANGUAGE_DIR = Path(__file__).resolve().parents[1] # going back by one parent

DEFAULT_TOKENIZER = LANGUAGE_DIR / "tokenizers" / "unigram_8000.model"
DEFAULT_SPLITS_DIR = LANGUAGE_DIR / "datasets" / "splits"
DEFAULT_OUTPUT_DIR = LANGUAGE_DIR / "datasets" / "tokenized"

SPLITS = ("train", "validation", "test")

def iter_text(jsonl_path: Path):
    """Yield text from a JSONL file, one line at a time."""
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            text = json.loads(line).get("text")
            text = text.strip()
            if text:
                yield text

def tokenize_split(
        split_name: str,
        input_path: Path,
        output_dir: Path,
        tokenizer: spm.SentencePieceProcessor,
        chunk_size: int,
        # overwrite: bool
) -> None:
    """Tokenize one split and save it as a uint16 binary token stream."""
    output_path = output_dir / f"{split_name}.bin"
    metadata_path = output_dir / f"{split_name}.json"
    partial_path = output_dir / f"{split_name}.bin.partial"

    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists."
        )

    total_tokens = 0
    total_documents = 0
    skipped_empty_encodings = 0
    token_buffer: list[int] = []

    from tqdm import tqdm

    progress = tqdm(
            iter_text(input_path),
            desc=f"Tokenizing {split_name}",
            unit="doc",
    )

    with partial_path.open("wb") as output_file:
        for document in progress:
            token_ids = tokenizer.encode(
                document,
                out_type=int,
                add_bos=False,
                add_eos=False,
            )

            if not token_ids:
                skipped_empty_encodings += 1
                continue

            # Add EOS token at the end of the sequence.
            token_ids.append(tokenizer.eos_id())

            token_buffer.extend(token_ids)
            total_tokens += len(token_ids)
            total_documents += 1

            if len(token_buffer) >= chunk_size:
                output_file.write(np.array(token_buffer, dtype=np.uint16).tobytes())
                token_buffer.clear()

            progress.set_postfix(
                tokens=f"{total_tokens:,}",
                skipped=skipped_empty_encodings,
            )


        # Write any remaining tokens in the buffer to the output file.
        if token_buffer:
            output_file.write(np.array(token_buffer, dtype=np.uint16).tobytes())
            token_buffer.clear()

    partial_path.replace(output_path)

    metadata = {
        "split": split_name,
        "source": str(input_path),
        "tokenizer": str(DEFAULT_TOKENIZER),
        "vocab_size": tokenizer.vocab_size(),
        "dtype": "uint16",
        "num_tokens": total_tokens,
        "num_documents": total_documents,
        "skipped_empty_encodings": skipped_empty_encodings,
        "pad_id": tokenizer.pad_id(),
        "unk_id": tokenizer.unk_id(),
        "bos_id": tokenizer.bos_id(),
        "eos_id": tokenizer.eos_id(),
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
        f.write("\n")

    print(
        f"Finished {split_name}: {total_documents:,} documents, "
        f"{total_tokens:,} tokens"
    )
    print(f"  Tokens:   {output_path}")
    print(f"  Metadata: {metadata_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize Hindi JSONL splits for language-model training."
    )

    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help=f"SentencePiece .model file. Default: {DEFAULT_TOKENIZER}",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
        help=f"Directory containing train/validation/test JSONL. Default: {DEFAULT_SPLITS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for .bin and metadata files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=SPLITS,
        help="Splits to prepare. Default: train validation test",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of buffered tokens before writing. Default: 1000000",
    )
    # parser.add_argument(
    #     "--overwrite",
    #     action="store_true",
    #     help="Replace existing output files.",
    # )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))

    if tokenizer.eos_id() != 3:
        raise ValueError(f"Expected EOS token ID to be 3. but got {tokenizer.eos_id()}")

    for split_name in args.splits:
        input_path = args.splits_dir / f"{split_name}.jsonl"

        if not input_path.is_file():
            raise FileNotFoundError(f"Split not found: {input_path}")

        tokenize_split(
            split_name=split_name,
            input_path=input_path,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            chunk_size=args.chunk_size,
            # overwrite=args.overwrite,
        )

if __name__ == "__main__":
    main()