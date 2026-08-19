from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    from .basic_doc_filtering import DocumentLevelFilter
    from .deduplication import Deduplicator
    from .lang_doc_filtering import LanguageIdentifier
    from .perplexity_filter import PerplexityFilter
    from .quality_filtering import CharacterQualityFilter, GibberishFilter
    from .various_normalization import TextNormalizer, UnicodeNormalizer
except ImportError:  # pragma: no cover - direct script execution support
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from basic_doc_filtering import DocumentLevelFilter
    from deduplication import Deduplicator
    from lang_doc_filtering import LanguageIdentifier
    from perplexity_filter import PerplexityFilter
    from quality_filtering import CharacterQualityFilter, GibberishFilter
    from various_normalization import TextNormalizer, UnicodeNormalizer


@dataclass
class PreprocessingConfig:
    min_word_count: int = 100
    min_devanagari_ratio: float = 0.2
    min_hindi_language_confidence: float = 0.6
    min_hindi_script_ratio: float = 0.15
    max_weird_char_ratio: float = 0.1
    max_repeated_char_ratio: float = 0.02
    preserve_paragraphs: bool = True
    normalize_unicode: bool = True
    convert_devanagari_digits: bool = True
    use_fasttext: bool = True
    use_pycld3: bool = True
    use_near_dedup: bool = True
    use_perplexity_filter: bool = False
    minhash_num_perm: int = 128
    minhash_threshold: float = 0.9
    text_fields: Sequence[str] = (
        "text",
        "content",
        "body",
        "article",
        "page_text",
        "clean_text",
        "document",
        "description",
        "title",
    )


@dataclass
class PipelineStats:
    total_records: int = 0
    kept_records: int = 0
    removed_records: int = 0
    malformed_json: int = 0
    empty_records: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)

    def add_reason(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def mark_removed(self, reason: str) -> None:
        self.removed_records += 1
        self.add_reason(reason)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "kept_records": self.kept_records,
            "removed_records": self.removed_records,
            "malformed_json": self.malformed_json,
            "empty_records": self.empty_records,
            "reasons": dict(sorted(self.reasons.items())),
        }


class HindiCorpusPipeline:
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.stats = PipelineStats()

        self.document_filter = DocumentLevelFilter(
            min_word_count=self.config.min_word_count,
        )
        self.unicode_normalizer = UnicodeNormalizer(normalize_unicode=self.config.normalize_unicode)
        self.language_identifier = LanguageIdentifier(
            use_fasttext=self.config.use_fasttext,
            use_pycld3=self.config.use_pycld3,
            min_hindi_language_confidence=self.config.min_hindi_language_confidence,
            min_hindi_script_ratio=self.config.min_hindi_script_ratio,
        )
        self.gibberish_filter = GibberishFilter()
        self.character_filter = CharacterQualityFilter(
            max_weird_char_ratio=self.config.max_weird_char_ratio,
            max_repeated_char_ratio=self.config.max_repeated_char_ratio,
        )
        self.text_normalizer = TextNormalizer(
            preserve_paragraphs=self.config.preserve_paragraphs,
            convert_devanagari_digits=self.config.convert_devanagari_digits,
        )
        self.deduplicator = Deduplicator(
            use_near_dedup=self.config.use_near_dedup,
            minhash_num_perm=self.config.minhash_num_perm,
            minhash_threshold=self.config.minhash_threshold,
        )
        self.perplexity_filter = PerplexityFilter(enabled=self.config.use_perplexity_filter)

    def reset_stats(self) -> None:
        self.stats = PipelineStats()

    def extract_text(self, record: Dict[str, Any]) -> Optional[str]:
        if not isinstance(record, dict):
            return None

        for field in self.config.text_fields:
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                return value
        return None

    def process_document(self, text: str) -> tuple[Optional[str], Optional[str]]:
        if not text or not isinstance(text, str):
            return None, "empty_text"

        stripped = text.strip()
        if not stripped:
            return None, "empty_text"

        if self.document_filter.short_document(stripped):
            return None, "short_document"

        if self.document_filter.garbage_document(stripped):
            return None, "garbage_document"

        text = self.unicode_normalizer.process(stripped)
        text = self.text_normalizer.normalize_whitespace(text)

        if self.deduplicator.exact_duplicate(text):
            return None, "exact_duplicate"

        if not self.language_identifier.is_hindi_text(text):
            return None, "non_hindi"

        if self.language_identifier.devanagari_ratio(text) < self.config.min_devanagari_ratio:
            return None, "low_devanagari_ratio"

        if self.gibberish_filter.is_gibberish(text):
            return None, "gibberish"

        if self.character_filter.is_low_quality(text):
            return None, "low_quality_characters"

        text = self.text_normalizer.remove_emojis(text)
        text = self.text_normalizer.mask_urls_and_emails(text)
        text = self.text_normalizer.normalize_punctuation_and_digits(text)

        if self.perplexity_filter.is_low_quality_by_perplexity(text):
            return None, "perplexity_filter"

        if self.deduplicator.near_duplicate(text):
            return None, "near_duplicate"

        return text.strip(), "passed"

    def process_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = self.extract_text(record)
        if text is None:
            return None

        cleaned_text, reason = self.process_document(text)
        if cleaned_text is None:
            return None

        updated = dict(record)
        for field in self.config.text_fields:
            if field in updated:
                updated[field] = cleaned_text
                break
        return updated

    def process_record_with_stats(self, record: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        text = self.extract_text(record)
        if text is None:
            return None, "missing_text_field"

        cleaned_text, reason = self.process_document(text)
        if cleaned_text is None:
            return None, reason

        updated = dict(record)
        for field in self.config.text_fields:
            if field in updated:
                updated[field] = cleaned_text
                break
        return updated, reason

    def process_jsonl(
        self,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        show_progress: bool = True,
        progress_desc: str = "Processing records",
    ) -> PipelineStats:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stream = input_path.open("r", encoding="utf-8")
        iterate = tqdm(stream, desc=progress_desc, unit="rec") if show_progress and tqdm is not None else stream

        try:
            with output_path.open("w", encoding="utf-8") as dst:
                for raw_line in iterate:
                    self.stats.total_records += 1
                    stripped_line = raw_line.strip()
                    if not stripped_line:
                        self.stats.empty_records += 1
                        self.stats.mark_removed("empty_record")
                        continue

                    try:
                        record = json.loads(stripped_line)
                    except json.JSONDecodeError:
                        self.stats.malformed_json += 1
                        self.stats.mark_removed("malformed_json")
                        continue

                    processed, reason = self.process_record_with_stats(record)
                    if processed is None:
                        self.stats.mark_removed(reason or "unknown_reason")
                        continue

                    dst.write(json.dumps(processed, ensure_ascii=False) + "\n")
                    self.stats.kept_records += 1
        finally:
            if iterate is not stream:
                iterate.close()
            stream.close()

        return self.stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hindi corpus preprocessing pipeline for Common Crawl JSONL data.")
    parser.add_argument("--input", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to the cleaned output JSONL file.")
    parser.add_argument("--min-words", type=int, default=100, help="Minimum word count for a valid document.")
    parser.add_argument("--min-dev-ratio", type=float, default=0.2, help="Minimum Devanagari script ratio.")
    parser.add_argument("--fasttext-model", default=None, help="Optional path to FastText LID model.")
    parser.add_argument("--no-fasttext", action="store_true", help="Disable FastText LID.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = PreprocessingConfig(
        min_word_count=args.min_words,
        min_devanagari_ratio=args.min_dev_ratio,
        use_fasttext=not args.no_fasttext,
    )

    if args.fasttext_model:
        os.environ["FASTTEXT_LID_MODEL"] = args.fasttext_model

    pipeline = HindiCorpusPipeline(config=config)
    stats = pipeline.process_jsonl(args.input, args.output, show_progress=not args.no_progress)
    print(f"Wrote {stats.kept_records} cleaned records to {args.output}")
    print("Removal stats:")
    for reason, count in stats.summary()["reasons"].items():
        print(f"  - {reason}: {count}")
    print(f"Total processed: {stats.total_records}")
    print(f"Malformed JSON: {stats.malformed_json}")
    print(f"Empty records: {stats.empty_records}")


if __name__ == "__main__":
    main()
