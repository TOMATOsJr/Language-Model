# import re
# import unicodedata

# try:
#     import ftfy
# except ImportError:  # pragma: no cover
#     ftfy = None

# try:
#     # from indicnlp.normalize import indic_normalize
#     from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
# except ImportError:  # pragma: no cover
#     indic_normalize = None


# class UnicodeNormalizer:
#     def __init__(self, normalize_unicode: bool = True):
#         self.normalize_unicode = normalize_unicode

#     def process(self, text: str) -> str:
#         if not text:
#             return text

#         normalized = unicodedata.normalize("NFC", text)
#         if self.normalize_unicode and ftfy is not None:
#             normalized = ftfy.fix_text(normalized)
#         return normalized


# class TextNormalizer:
#     def __init__(self, preserve_paragraphs: bool = True, convert_devanagari_digits: bool = True):
#         self.preserve_paragraphs = preserve_paragraphs
#         self.convert_devanagari_digits = convert_devanagari_digits

#     def normalize_whitespace(self, text: str) -> str:
#         text = text.replace("\u00A0", " ")
#         text = text.replace("\r\n", "\n").replace("\r", "\n")
#         text = re.sub(r"[ \t]+", " ", text)
#         if self.preserve_paragraphs:
#             text = re.sub(r"\n{3,}", "\n\n", text)
#             text = re.sub(r" *\n *", "\n", text)
#             text = re.sub(r"\n{2,}", "\n\n", text)
#         else:
#             text = re.sub(r"\s+", " ", text)
#         return text.strip()

#     def normalize_punctuation_and_digits(self, text: str) -> str:
#         if indic_normalize is not None:
#             try:
#                 text = indic_normalize.indic_normalize.DevanagariNormalizer().normalize(text)
#             except AttributeError:
#                 text = indic_normalize.indic_normalize(text)
#         else:
#             text = text.replace("||", "॥").replace("|", "।")

#         if self.convert_devanagari_digits:
#             mapping = str.maketrans(
#                 {
#                     "०": "0",
#                     "१": "1",
#                     "२": "2",
#                     "३": "3",
#                     "४": "4",
#                     "५": "5",
#                     "६": "6",
#                     "७": "7",
#                     "८": "8",
#                     "९": "9",
#                 }
#             )
#             text = text.translate(mapping)
#         return text

#     def mask_urls_and_emails(self, text: str) -> str:
#         text = re.sub(r"https?://\S+|www\.\S+", " <URL> ", text, flags=re.IGNORECASE)
#         text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " <EMAIL> ", text)
#         return text

import re
import unicodedata

try:
    import ftfy
except ImportError:  # pragma: no cover
    ftfy = None

try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
except ImportError:  # pragma: no cover
    IndicNormalizerFactory = None


class UnicodeNormalizer:
    def __init__(self, normalize_unicode: bool = True):
        self.normalize_unicode = normalize_unicode

        self.indic_normalizer = None

        if IndicNormalizerFactory is not None:
            factory = IndicNormalizerFactory()
            self.indic_normalizer = factory.get_normalizer("hi")

    def process(self, text: str) -> str:
        if not text:
            return text

        normalized = text

        # Unicode canonical normalization
        if self.normalize_unicode:
            normalized = unicodedata.normalize(
                "NFC",
                normalized,
            )

        # Fix common Unicode mojibake / broken text
        if ftfy is not None:
            normalized = ftfy.fix_text(normalized)

        # Indic NLP Hindi normalization
        if self.indic_normalizer is not None:
            normalized = self.indic_normalizer.normalize(
                normalized
            )

        return normalized


class TextNormalizer:
    def __init__(
        self,
        preserve_paragraphs: bool = True,
        convert_devanagari_digits: bool = True,
    ):
        self.preserve_paragraphs = preserve_paragraphs
        self.convert_devanagari_digits = convert_devanagari_digits

    def normalize_whitespace(self, text: str) -> str:
        text = text.replace("\u00A0", " ")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)

        if self.preserve_paragraphs:
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" *\n *", "\n", text)
            text = re.sub(r"\n{2,}", "\n\n", text)
        else:
            text = re.sub(r"\s+", " ", text)

        return text.strip()

    def normalize_punctuation_and_digits(self, text: str) -> str:
        # Devanagari danda normalization
        text = text.replace("||", "॥")
        text = text.replace("|", "।")

        if self.convert_devanagari_digits:
            mapping = str.maketrans(
                {
                    "०": "0",
                    "१": "1",
                    "२": "2",
                    "३": "3",
                    "४": "4",
                    "५": "5",
                    "६": "6",
                    "७": "7",
                    "८": "8",
                    "९": "9",
                }
            )

            text = text.translate(mapping)

        return text

    def mask_urls_and_emails(self, text: str) -> str:
        text = re.sub(
            r"https?://\S+|www\.\S+",
            " <URL> ",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
            " <EMAIL> ",
            text,
        )

        return text

    def remove_emojis(self, text: str) -> str:
        emoji_pattern = re.compile(
            "["
            "\U0001F1E6-\U0001F1FF"  # flags
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F680-\U0001F6FF"  # transport
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0001FA00-\U0001FAFF"  # symbols
            "\U00002700-\U000027BF"  # dingbats
            "\U00002600-\U000026FF"  # miscellaneous symbols
            "\U00002300-\U000023FF"
            "\uFE0F"                 # variation selector
            "\u200D"                 # zero-width joiner
            "]+",
            flags=re.UNICODE,
        )

        return emoji_pattern.sub("", text)
