"""Reject candidates that are translations rather than romanizations.

:mod:`gazetteer.plausibility` catches pairs whose *lengths* cannot correspond.
It cannot catch a pair whose length is perfectly ordinary and whose meaning is
simply the wrong relationship. Wikidata labels an entity in every language it
has, so an unscoped pull yields plenty of these:

======================  ===========  ==============
key                     harvested    mechanical
======================  ===========  ==============
``उड़ान``                ``flight``   ``urana``
``क्षेत्रों``              ``of``       ``ksetrom``
``कब्र``                 ``forbes``   ``kabra``
``आलोचकों``              ``editors``  ``alocakom``
======================  ===========  ==============

The test is whether the candidate resembles a *mechanical* romanization of the
key. That mechanical form comes from ``indic_transliteration``'s ISO/IAST
transliterator -- a rule table, not a model, so nothing the corpus is later
scored against is consulted and no circularity is introduced. Similarity is
:class:`difflib.SequenceMatcher` on the diacritic-stripped forms, which is
lenient enough for real spelling variation (``गायत्री`` -> ``gayathri`` scores
0.93 against ``gayatri``) and unforgiving of unrelated words.

**The threshold is measured.** Over the Hindi harvest, taking the 2,072 pairs on
which at least two independent trust groups agree as the reference set:

=========  ==========
percentile similarity
=========  ==========
p1          0.29
p5          0.57
p10         0.67
p25         0.80
p50         0.89
=========  ==========

Cutting at p1 gives **0.30**, which rejects 1.0% of agreed pairs against 5.7% of
single-source ones -- the enrichment is the point, since single-source Wikidata
rows are where the translations live. A stricter cut buys little: 0.50 rejects
twice as many agreed pairs to reach only 9.3% of single-source ones.

This filter needs the source language, because the mechanical transliterator
needs to know which script it is reading.
"""

from __future__ import annotations

import difflib
from functools import lru_cache

from indicate.normalize import latin_form

#: Similarity at or above which a candidate is accepted, from the measurement
#: in the module docstring (the 1st percentile of independently agreed pairs).
MIN_SIMILARITY = 0.30

#: ``indic_transliteration`` scheme name for each corpus language.
LANGUAGE_SCHEME = {
    "assamese": "bengali",
    "bengali": "bengali",
    "gujarati": "gujarati",
    "hindi": "devanagari",
    "kannada": "kannada",
    "malayalam": "malayalam",
    "marathi": "devanagari",
    "odia": "oriya",
    "punjabi": "gurmukhi",
    "tamil": "tamil",
    "telugu": "telugu",
}


@lru_cache(maxsize=1 << 16)
def mechanical_romanization(key: str, lang: str) -> str:
    """Romanize ``key`` by rule, with diacritics stripped.

    Args:
        key: A token in an Indic script.
        lang: Corpus language name, which selects the source script.

    Returns:
        The rule-based romanization, or ``""`` when this language has no scheme
        or the transliterator cannot read the token.
    """
    scheme = LANGUAGE_SCHEME.get(lang)
    if not scheme or not key:
        return ""
    from indic_transliteration import sanscript

    try:
        return latin_form(sanscript.transliterate(key, scheme, sanscript.IAST))
    except Exception:  # pragma: no cover - malformed input in a source file
        return ""


def phonetic_similarity(key: str, latin: str, lang: str) -> float:
    """Score how much ``latin`` looks like a romanization of ``key``.

    Args:
        key: A token in an Indic script.
        latin: Candidate romanization, already in :func:`latin_form`.
        lang: Corpus language name.

    Returns:
        A ratio in ``[0, 1]``; ``1.0`` when no mechanical form is available, so
        an unsupported language is never filtered on a number it cannot compute.
    """
    mechanical = mechanical_romanization(key, lang)
    if not mechanical or not latin:
        return 1.0
    return difflib.SequenceMatcher(None, mechanical, latin).ratio()


def is_phonetic_match(
    key: str, latin: str, lang: str, *, threshold: float = MIN_SIMILARITY
) -> bool:
    """Report whether a candidate is close enough to be a romanization.

    Args:
        key: A token in an Indic script.
        latin: Candidate romanization.
        lang: Corpus language name.
        threshold: Minimum similarity to accept.

    Returns:
        True when the candidate resembles a mechanical romanization of the key.
    """
    return phonetic_similarity(key, latin, lang) >= threshold
