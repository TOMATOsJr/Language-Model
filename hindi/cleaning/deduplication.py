# import re
# import sys
# from pathlib import Path
# from typing import List, Optional

# try:
#     from datasketch import MinHash, MinHashLSH
# except ImportError:  # pragma: no cover
#     MinHash = None
#     MinHashLSH = None

# try:
#     from .basic_doc_filtering import DocumentLevelFilter
# except ImportError:  # pragma: no cover
#     sys.path.insert(0, str(Path(__file__).resolve().parent))
#     from basic_doc_filtering import DocumentLevelFilter


# class Deduplicator:
#     def __init__(
#         self,
#         exact_dedup_cache_size: int = 200000,
#         use_near_dedup: bool = False,
#         minhash_num_perm: int = 128,
#         minhash_threshold: float = 0.9,
#     ):
#         self.exact_seen: set[str] = set()
#         self.exact_dedup_cache_size = exact_dedup_cache_size
#         self.use_near_dedup = use_near_dedup
#         self.minhash_num_perm = minhash_num_perm
#         self.minhash_threshold = minhash_threshold
#         self.minhash_lsh = None

#         if self.use_near_dedup and MinHash is not None and MinHashLSH is not None:
#             self.minhash_lsh = MinHashLSH(threshold=self.minhash_threshold, num_perm=self.minhash_num_perm)

#     def exact_duplicate(self, text: str) -> bool:
#         digest = DocumentLevelFilter.document_hash(text)
#         if digest in self.exact_seen:
#             return True
#         if len(self.exact_seen) >= self.exact_dedup_cache_size:
#             self.exact_seen.clear()
#         self.exact_seen.add(digest)
#         return False

#     def near_duplicate(self, text: str) -> bool:
#         if self.minhash_lsh is None:
#             return False

#         tokens = self._tokenize(text)
#         if not tokens:
#             return False

#         m = MinHash(num_perm=self.minhash_num_perm)
#         for token in tokens:
#             m.update(token.encode("utf-8"))

#         if self.minhash_lsh.query(m) is not None:
#             return True

#         key = self._minhash_key(text)
#         if key is None:
#             return False

#         self.minhash_lsh.insert(key, m)
#         return False

#     @staticmethod
#     def _tokenize(text: str) -> List[str]:
#         tokens = re.findall(r"\b[\w\u0900-\u097F]+\b", text, flags=re.UNICODE)
#         return [token.lower() for token in tokens if token.strip()]

#     @staticmethod
#     def _minhash_key(text: str) -> Optional[str]:
#         if not text:
#             return None
#         return DocumentLevelFilter.document_hash(text)

import re
import sys
from pathlib import Path
from typing import List

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:  # pragma: no cover
    MinHash = None
    MinHashLSH = None

try:
    from .basic_doc_filtering import DocumentLevelFilter
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from basic_doc_filtering import DocumentLevelFilter


class Deduplicator:
    """
    Exact and near-duplicate detection for Hindi documents.

    Assumption:
        Unicode normalization and whitespace normalization have
        already been performed by the pipeline before this class
        receives the text.

    Exact deduplication:
        Global in-memory hash set.

    Near deduplication:
        MinHash + LSH over word-level n-grams.
    """

    def __init__(
        self,
        use_near_dedup: bool = False,
        minhash_num_perm: int = 128,
        minhash_threshold: float = 0.90,
        shingle_size: int = 3,
    ):
        self.use_near_dedup = use_near_dedup
        self.minhash_num_perm = minhash_num_perm
        self.minhash_threshold = minhash_threshold
        self.shingle_size = shingle_size

        # 410k records is small enough for global exact deduplication.
        self.exact_seen: set[str] = set()

        self.minhash_lsh = None

        if self.use_near_dedup:
            if MinHash is None or MinHashLSH is None:
                raise ImportError(
                    "datasketch is required when near deduplication "
                    "is enabled."
                )

            self.minhash_lsh = MinHashLSH(
                threshold=self.minhash_threshold,
                num_perm=self.minhash_num_perm,
            )

    # ============================================================
    # Exact deduplication
    # ============================================================

    def exact_duplicate(self, text: str) -> bool:
        """
        Return True if this text has already been seen.

        The input is assumed to have already undergone Unicode
        and whitespace normalization in the main pipeline.
        """

        if not text or not text.strip():
            return True

        digest = DocumentLevelFilter.document_hash(text)

        if digest in self.exact_seen:
            return True

        self.exact_seen.add(digest)

        return False

    # ============================================================
    # Near deduplication
    # ============================================================

    def near_duplicate(self, text: str) -> bool:
        """
        Return True if the document is a near duplicate of a
        previously accepted document.

        Uses MinHash over word-level shingles.
        """

        if self.minhash_lsh is None:
            return False

        tokens = self._tokenize(text)

        if len(tokens) < self.shingle_size:
            return False

        shingles = self._make_shingles(
            tokens,
            self.shingle_size,
        )

        if not shingles:
            return False

        minhash = MinHash(
            num_perm=self.minhash_num_perm
        )

        for shingle in shingles:
            minhash.update(
                shingle.encode("utf-8")
            )

        # query() returns a list of matching document keys.
        matches = self.minhash_lsh.query(minhash)

        if matches:
            return True

        # Text has not been seen before, so add it to the LSH index.
        key = DocumentLevelFilter.document_hash(text)

        self.minhash_lsh.insert(key, minhash)

        return False

    # ============================================================
    # Tokenization for near deduplication
    # ============================================================

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize text into words for MinHash.

        This is NOT the model tokenizer.

        It keeps:
            - Devanagari words
            - Latin words
            - numbers/alphanumeric tokens
        """

        tokens = re.findall(
            r"[\w\u0900-\u097F]+",
            text,
            flags=re.UNICODE,
        )

        return [
            token.lower()
            for token in tokens
            if token.strip()
        ]

    # ============================================================
    # Word-level shingles
    # ============================================================

    @staticmethod
    def _make_shingles(
        tokens: List[str],
        size: int,
    ) -> set[str]:
        """
        Create word-level n-grams.

        Example, size=3:

            भारत सरकार ने नई शिक्षा नीति

        becomes:

            भारत सरकार ने
            सरकार ने नई
            ने नई शिक्षा
            नई शिक्षा नीति
        """

        if len(tokens) < size:
            return set()

        return {
            " ".join(tokens[i:i + size])
            for i in range(len(tokens) - size + 1)
        }