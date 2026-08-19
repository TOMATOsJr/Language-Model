# import hashlib
# import re
# from typing import Iterable, Optional

# try:
#     import blake3
# except ImportError:  # pragma: no cover
#     blake3 = None


# DEFAULT_GARBAGE_PATTERNS: Iterable[re.Pattern[str]] = (
#     re.compile(r"404\s+not\s+found", re.IGNORECASE),
#     re.compile(r"access\s+denied", re.IGNORECASE),
#     re.compile(r"javascript\s+is\s+disabled", re.IGNORECASE),
#     re.compile(r"\bpage\s+not\s+found\b", re.IGNORECASE),
#     re.compile(r"\bforbidden\b", re.IGNORECASE),
# )


# class DocumentLevelFilter:
#     def __init__(self, min_word_count: int = 100, garbage_patterns: Optional[Iterable[re.Pattern[str]]] = None):
#         self.min_word_count = min_word_count
#         self.garbage_patterns = tuple(garbage_patterns or DEFAULT_GARBAGE_PATTERNS)

#     def short_document(self, text: str) -> bool:
#         return len(text.split()) < self.min_word_count

#     def garbage_document(self, text: str) -> bool:
#         return any(pattern.search(text) for pattern in self.garbage_patterns)

#     @staticmethod
#     def document_hash(text: str) -> str:
#         raw = text.encode("utf-8")
#         if blake3 is not None:
#             return blake3.blake3(raw).hexdigest()
#         return hashlib.sha256(raw).hexdigest()

#     def filter_document(self, text: str) -> bool:
#         return self.short_document(text) or self.garbage_document(text)

import hashlib
import re
from typing import Iterable, Optional

try:
    import blake3
except ImportError:  # pragma: no cover
    blake3 = None


DEFAULT_GARBAGE_PATTERNS: Iterable[re.Pattern[str]] = (
    re.compile(r"\b404\s+not\s+found\b", re.IGNORECASE),
    re.compile(r"\baccess\s+denied\b", re.IGNORECASE),
    re.compile(r"\bjavascript\s+is\s+disabled\b", re.IGNORECASE),
    re.compile(r"\bpage\s+not\s+found\b", re.IGNORECASE),
)


class DocumentLevelFilter:
    """
    Performs cheap document-level filtering before more expensive
    language, quality, perplexity, and deduplication checks.
    """

    def __init__(
        self,
        min_word_count: int = 100,
        garbage_patterns: Optional[
            Iterable[re.Pattern[str]]
        ] = None,
    ):
        self.min_word_count = min_word_count

        self.garbage_patterns = tuple(
            garbage_patterns
            if garbage_patterns is not None
            else DEFAULT_GARBAGE_PATTERNS
        )

    # ============================================================
    # Length
    # ============================================================

    def short_document(self, text: str) -> bool:
        """
        Return True if the document contains fewer than the
        minimum number of whitespace-separated words.
        """

        if not text or not text.strip():
            return True

        return len(text.split()) < self.min_word_count

    # ============================================================
    # Garbage / error pages
    # ============================================================

    def garbage_document(self, text: str) -> bool:
        """
        Return True if the document contains a strong indicator
        of a web error/garbage page.
        """

        if not text:
            return True

        return any(
            pattern.search(text)
            for pattern in self.garbage_patterns
        )

    # ============================================================
    # Hashing
    # ============================================================

    @staticmethod
    def document_hash(text: str) -> str:
        """
        Generate a deterministic hash for deduplication.

        The caller is responsible for any normalization required
        before hashing.
        """

        raw = text.encode("utf-8")

        if blake3 is not None:
            return blake3.blake3(raw).hexdigest()

        return hashlib.sha256(raw).hexdigest()

    # ============================================================
    # Main filter
    # ============================================================

    def filter_document(self, text: str) -> bool:
        """
        Return True if the document should be rejected.
        """

        if self.short_document(text):
            return True

        if self.garbage_document(text):
            return True

        return False
