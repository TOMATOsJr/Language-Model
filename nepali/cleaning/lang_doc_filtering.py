# import os

# try:
#     import fasttext
# except ImportError:  # pragma: no cover
#     fasttext = None

# try:
#     import pycld3
# except ImportError:  # pragma: no cover
#     pycld3 = None


# class LanguageIdentifier:
#     def __init__(
#         self,
#         use_fasttext: bool = True,
#         use_pycld3: bool = True,
#         min_nepali_language_confidence: float = 0.6,
#         min_nepali_script_ratio: float = 0.15,
#     ):
#         self.use_fasttext = use_fasttext
#         self.use_pycld3 = use_pycld3
#         self.min_nepali_language_confidence = min_nepali_language_confidence
#         self.min_nepali_script_ratio = min_nepali_script_ratio
#         self.fasttext_model = None
#         self._load_fasttext_model()

#     def _load_fasttext_model(self) -> None:
#         if not self.use_fasttext or fasttext is None:
#             return

#         model_path = os.environ.get("FASTTEXT_LID_MODEL")
#         if model_path and os.path.exists(model_path):
#             self.fasttext_model = fasttext.load_model(model_path)

#     def is_nepali_text(self, text: str) -> bool:
#         if not text.strip():
#             return False

#         if self.fasttext_model is not None:
#             prediction = self.fasttext_model.predict(text.replace("\n", " "), k=3)
#             labels = prediction[0]
#             scores = prediction[1]
#             for label, score in zip(labels, scores):
#                 label_name = label.replace("__label__", "")
#                 if label_name.startswith("ne") or "nepali" in label_name.lower():
#                     return score >= self.min_nepali_language_confidence
#             return max(scores) >= self.min_nepali_language_confidence

#         if pycld3 is not None and self.use_pycld3:
#             result = pycld3.get_language(text)
#             if result is not None and result.language == "ne":
#                 return result.probability >= self.min_nepali_language_confidence

#         return self.devanagari_ratio(text) >= self.min_nepali_script_ratio

#     @staticmethod
#     def devanagari_ratio(text: str) -> float:
#         if not text:
#             return 0.0
#         dev_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
#         return dev_chars / len(text)

import numpy as np

# =========================================================
# Monkey-patch NumPy 2.0+ compatibility issue with fastText
# =========================================================
_orig_np_array = np.array
np.array = lambda obj, *args, **kwargs: _orig_np_array(
    obj, *args, **{k: v for k, v in kwargs.items() if k != 'copy'}
)


import os
from typing import Optional

try:
    import fasttext
except ImportError:  # pragma: no cover
    fasttext = None

try:
    import pycld3
except ImportError:  # pragma: no cover
    pycld3 = None


class LanguageIdentifier:
    """
    Identifies Nepali text using:
        1. FastText language identification (preferred)
        2. pycld3 (fallback)
        3. Devanagari script ratio (last-resort fallback)

    Note:
        Devanagari ratio alone cannot distinguish Nepali from Hindi,
        Marathi, Sanskrit, etc. It is only a script-based fallback.
    """

    def __init__(
        self,
        use_fasttext: bool = True,
        use_pycld3: bool = True,
        min_nepali_language_confidence: float = 0.6,
        min_nepali_script_ratio: float = 0.15,
        fasttext_max_chars: int = 5000,
    ):
        self.use_fasttext = use_fasttext
        self.use_pycld3 = use_pycld3
        self.min_nepali_language_confidence = (
            min_nepali_language_confidence
        )
        self.min_nepali_script_ratio = min_nepali_script_ratio
        self.fasttext_max_chars = fasttext_max_chars

        self.fasttext_model = None

        self._load_fasttext_model()

    # ============================================================
    # FastText
    # ============================================================

    def _load_fasttext_model(self) -> None:
        """Load the FastText language identification model."""
        if not self.use_fasttext or fasttext is None:
            return

        model_path = os.environ.get("FASTTEXT_LID_MODEL")

        if not model_path:
            return

        if not os.path.exists(model_path):
            return

        self.fasttext_model = fasttext.load_model(model_path)

    # ============================================================
    # Main API
    # ============================================================

    def is_nepali_text(self, text: str) -> bool:
        """
        Return True if the text is likely Nepali.

        Priority:
            FastText -> pycld3 -> Devanagari heuristic
        """

        if not text or not text.strip():
            return False

        # --------------------------------------------------------
        # 1. FastText
        # --------------------------------------------------------

        if self.fasttext_model is not None:
            return self._is_nepali_fasttext(text)
        else:
            raise RuntimeError(
                "FastText model is not loaded. "
                "Nepali language identification cannot proceed."
            )

        # --------------------------------------------------------
        # 2. pycld3
        # --------------------------------------------------------

        if self.use_pycld3 and pycld3 is not None:
            result = pycld3.get_language(text)

            if result is not None:
                return (
                    result.language == "ne"
                    and result.probability
                    >= self.min_nepali_language_confidence
                )

        # --------------------------------------------------------
        # 3. Script-based fallback
        # --------------------------------------------------------
        #
        # IMPORTANT:
        # This does NOT prove that the text is Nepali.
        # It only says that enough alphabetic characters are
        # written in Devanagari.
        # --------------------------------------------------------

        return (
            self.devanagari_ratio(text)
            >= self.min_nepali_script_ratio
        )

    # ============================================================
    # FastText classification
    # ============================================================

    def _is_nepali_fasttext(self, text: str) -> bool:
        """Classify text using FastText."""

        sample = self._make_sample(text)

        labels, scores = self.fasttext_model.predict(
            sample,
            k=3,
        )

        for label, score in zip(labels, scores):
            label_name = label.replace("__label__", "")

            if label_name == "ne":
                return score >= self.min_nepali_language_confidence

        # IMPORTANT:
        # Do NOT use max(scores) here.
        #
        # If FastText predicts:
        #   en = 0.95
        #   fr = 0.03
        #   de = 0.02
        #
        # the document must NOT be classified as Nepali.
        return False

    # ============================================================
    # Text sampling
    # ============================================================

    def _make_sample(self, text: str) -> str:
        """
        Create a representative sample for language identification.

        For short documents, use everything.

        For long documents, take text from the beginning, middle,
        and end so that classification does not depend entirely
        on the first few thousand characters.
        """

        text = text.replace("\n", " ").strip()

        if len(text) <= self.fasttext_max_chars:
            return text

        third = self.fasttext_max_chars // 3

        beginning = text[:third]

        middle_start = max(
            0,
            len(text) // 2 - third // 2,
        )
        middle = text[
            middle_start : middle_start + third
        ]

        ending = text[-third:]

        return f"{beginning} {middle} {ending}"

    # ============================================================
    # Script detection
    # ============================================================

    @staticmethod
    def devanagari_ratio(text: str) -> float:
        """
        Return the fraction of alphabetic characters that belong
        to the Devanagari Unicode block.

        This is a SCRIPT ratio, not a LANGUAGE ratio.

        Example:
            Nepali       -> high
            Nepali      -> high
            Marathi     -> high
            Sanskrit    -> high

        Therefore this function cannot distinguish those languages.
        """

        if not text:
            return 0.0

        alphabetic_chars = [
            ch for ch in text
            if ch.isalpha()
        ]

        if not alphabetic_chars:
            return 0.0

        devanagari_chars = sum(
            1
            for ch in alphabetic_chars
            if "\u0900" <= ch <= "\u097F"
        )

        return (
            devanagari_chars
            / len(alphabetic_chars)
        )
