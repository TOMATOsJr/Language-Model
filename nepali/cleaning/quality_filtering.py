import math
import re
import unicodedata
from typing import Tuple


class GibberishFilter:
    @staticmethod
    def shannon_entropy(text: str) -> float:
        if not text:
            return 0.0
        counts = {}
        for ch in text:
            counts[ch] = counts.get(ch, 0) + 1
        total = len(text)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    def is_gibberish(self, text: str) -> bool:
        if not text:
            return True

        sample = re.sub(r"\s+", "", text)
        if not sample:
            return True

        entropy = self.shannon_entropy(sample)
        if entropy < 2.0:
            return True

        alpha_ratio = sum(1 for ch in sample if ch.isalpha()) / max(len(sample), 1)
        if alpha_ratio < 0.45:
            return True

        return False

class CharacterQualityFilter:
    def __init__(
        self,
        max_weird_char_ratio: float = 0.10,
        max_repeated_char_ratio: float = 0.02,
        min_alpha_ratio: float = 0.20,
    ):
        self.max_weird_char_ratio = max_weird_char_ratio
        self.max_repeated_char_ratio = max_repeated_char_ratio
        self.min_alpha_ratio = min_alpha_ratio

    def quality_score(self, text: str) -> Tuple[float, float, float]:
        """
        Returns:
            alpha_ratio:       fraction of characters that are alphabetic
            weird_ratio:       fraction of characters that are unusual
            repeated_ratio:   fraction of characters belonging to
                              suspicious repeated-character runs
        """
        if not text:
            return 0.0, 0.0, 0.0

        total = len(text)

        # Alphabetic characters, including Devanagari.
        alphabetic = sum(
            1 for ch in text
            if ch.isalpha()
        )

        # Unicode symbols and control/format characters.
        #
        # Whitespace is excluded because spaces/newlines are normal.
        weird = sum(
            1
            for ch in text
            if (
                unicodedata.category(ch).startswith(("S", "C"))
                and not ch.isspace()
            )
        )

        # Count characters that belong to suspicious runs such as:
        #
        # !!!!!!
        # ........
        # aaaaaaa
        # ؟؟؟؟؟؟
        #
        # {4,} means at least 5 consecutive copies.
        repeated_chars = 0

        for match in re.finditer(r"(.)\1{4,}", text):
            repeated_chars += len(match.group())

        alpha_ratio = alphabetic / total
        weird_ratio = weird / total
        repeated_ratio = repeated_chars / total

        return alpha_ratio, weird_ratio, repeated_ratio

    def is_low_quality(self, text: str) -> bool:
        alpha_ratio, weird_ratio, repeated_ratio = self.quality_score(text)

        if weird_ratio > self.max_weird_char_ratio:
            return True

        if repeated_ratio > self.max_repeated_char_ratio:
            return True

        if alpha_ratio < self.min_alpha_ratio:
            return True

        return False