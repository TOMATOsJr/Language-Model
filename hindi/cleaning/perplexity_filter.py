# try:
#     import kenlm
# except ImportError:  # pragma: no cover
#     kenlm = None


# class PerplexityFilter:
#     def __init__(self, enabled: bool = False, model=None):
#         self.enabled = enabled
#         self.model = model

#     def is_low_quality_by_perplexity(self, text: str) -> bool:
#         if not self.enabled or kenlm is None:
#             return False
#         if self.model is None:
#             return False
#         try:
#             score = self.model.score(text)
#             return score < 0.0
#         except Exception:
#             return False

try:
    import kenlm
except ImportError:  # pragma: no cover
    kenlm = None


class PerplexityFilter:
    """
    Filters documents using a KenLM language model.

    Lower perplexity:
        More likely to resemble the language model's training data.

    Higher perplexity:
        Less likely / potentially noisy or out-of-domain.

    Important:
        The KenLM model should be trained on clean Hindi text if this
        filter is being used for a Hindi corpus.
    """

    def __init__(
        self,
        enabled: bool = False,
        model=None,
        min_perplexity: float = 0.0,
        max_perplexity: float = 1000.0,
    ):
        self.enabled = enabled
        self.model = model

        self.min_perplexity = min_perplexity
        self.max_perplexity = max_perplexity

    def perplexity(self, text: str) -> float | None:
        """
        Calculate document perplexity using KenLM.

        Returns None if perplexity cannot be calculated.
        """

        if not self.enabled:
            return None

        if kenlm is None:
            return None

        if self.model is None:
            return None

        if not text or not text.strip():
            return None

        try:
            # KenLM score is a log probability.
            log_score = self.model.score(
                text,
                bos=True,
                eos=True,
            )

            # KenLM scores words/tokens. Using split() is a
            # reasonable approximation when working with a
            # word-level Hindi n-gram model.
            num_tokens = len(text.split())

            if num_tokens == 0:
                return None

            # Convert log probability to perplexity.
            #
            # KenLM's default log base is usually log10, so:
            #
            # PPL = 10 ^ (-log10_probability / N)
            #
            # If your model was configured differently, adjust
            # accordingly.
            return 10 ** (-log_score / num_tokens)

        except Exception:
            return None

    def is_low_quality_by_perplexity(self, text: str) -> bool:
        """
        Return True if the document's perplexity falls outside
        the accepted range.
        """

        if not self.enabled:
            return False

        ppl = self.perplexity(text)

        if ppl is None:
            # Don't reject a document just because the optional
            # perplexity stage failed.
            return False

        if ppl < self.min_perplexity:
            return True

        if ppl > self.max_perplexity:
            return True

        return False