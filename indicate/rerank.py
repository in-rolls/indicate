from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


class Reranker:
    """Re-rank beam candidates by interpolating model score with a char LM.

    Following IndicXlit (Aksharantar, EMNLP Findings 2023), the top-k beam
    hypotheses are re-scored as ``F = alpha * model_score + (1 - alpha) * lm``.
    Here the LM is a character n-gram model over the training-side English
    romanizations (so it generalizes to unseen names, unlike a word unigram),
    with both scores length-normalized to keep them comparable.
    """

    def __init__(
        self, words: Iterable[str], alpha: float = 0.9, order: int = 3, k: float = 1.0
    ) -> None:
        self.alpha = alpha
        self.order = order
        self.k = k
        self._ngram: Counter[str] = Counter()
        self._context: Counter[str] = Counter()
        self._vocab: set[str] = set()
        for word in words:
            self._add(word)
        self._vocab_size = max(len(self._vocab), 1)

    def _grams(self, word: str) -> Iterable[tuple[str, str, str]]:
        padded = "^" * (self.order - 1) + word + "$"
        for i in range(self.order - 1, len(padded)):
            ctx = padded[i - self.order + 1 : i]
            yield ctx, padded[i], padded[i - self.order + 1 : i + 1]

    def _add(self, word: str) -> None:
        for ctx, ch, gram in self._grams(word):
            self._ngram[gram] += 1
            self._context[ctx] += 1
            self._vocab.add(ch)

    def lm_logprob(self, word: str) -> float:
        """Length-normalized char-n-gram log-probability (add-k smoothed)."""
        if not word:
            return -1e9
        total = 0.0
        n = 0
        for ctx, _, gram in self._grams(word):
            num = self._ngram.get(gram, 0) + self.k
            den = self._context.get(ctx, 0) + self.k * self._vocab_size
            total += math.log(num / den)
            n += 1
        return total / max(n, 1)

    def best(self, candidates: list[tuple[str, float]]) -> str:
        """Return the candidate text maximizing the interpolated score.

        ``candidates`` are ``(text, model_score)`` where model_score is the
        beam's length-normalized log-probability.
        """
        if not candidates:
            return ""
        best_text, best_score = candidates[0][0], float("-inf")
        for text, model_score in candidates:
            combined = self.alpha * model_score + (1 - self.alpha) * self.lm_logprob(
                text
            )
            if combined > best_score:
                best_text, best_score = text, combined
        return best_text
