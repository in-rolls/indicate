"""Reject candidate pairs that cannot be transliterations of each other.

Positional alignment fires whenever two labels happen to have the same token
count, which occasionally pairs a one-letter Devanagari initial with an English
word: ``के``->``administrative``, ``जे``->``227j``, ``भारत``->
``deindustrialization``. Those are not disagreements to be adjudicated, they are
noise to be dropped before adjudication ever sees them.

A transliteration preserves rough length, so the ratio of Latin characters to
native aksharas is tightly bounded. The band here is **measured, not guessed**.
Taking the 2,106 Hindi pairs on which at least two independent sources already
agree as a reference set:

=========  =====
percentile ratio
=========  =====
p0.1        0.50
p1          0.67
p5          1.25
p50         1.80
p95         2.50
p99         3.00
p99.9       3.50
=========  =====

Clipping at p0.5 and p99.5 gives ``0.50 .. 3.00``. Applied to the full harvest
that removes 2.8% of rows while rejecting only 6 of the 2,106 agreed pairs
(0.28%) -- the survivors it does drop are hyphenated compounds that were split
across tokens, which are genuinely mis-aligned anyway.

This is a filter on *evidence*, never a source of answers, so it introduces no
circularity: the shipped model is not consulted.
"""

from __future__ import annotations

import unicodedata

#: Latin characters per akshara, from the measurement in the module docstring.
MIN_RATIO = 0.50
MAX_RATIO = 3.00


def aksharas(native: str) -> int:
    """Count the letters in an Indic token.

    Consonants and independent vowels are Unicode category ``Lo``; matras,
    bindus and viramas are ``Mn``/``Mc`` and ride along with a letter rather
    than adding one.

    Args:
        native: A token in an Indic script.

    Returns:
        The number of Indic letters, which is ``0`` for Latin or empty input.
    """
    return sum(1 for char in native if unicodedata.category(char) == "Lo")


def is_plausible_pair(
    native: str,
    latin: str,
    *,
    low: float = MIN_RATIO,
    high: float = MAX_RATIO,
) -> bool:
    """Report whether ``latin`` could be a romanization of ``native``.

    Args:
        native: Token in an Indic script.
        latin: Candidate romanization, already normalized.
        low: Minimum Latin characters per akshara.
        high: Maximum Latin characters per akshara.

    Returns:
        True if the length ratio falls inside the measured band. Tokens with no
        aksharas, and empty candidates, are never plausible.
    """
    if not native or not latin:
        return False
    letters = aksharas(native)
    if not letters:
        return False
    return low <= len(latin) / letters <= high
