"""Fast script detection for corpus tokens.

``EDGE_NOISE`` and ``strip_edge_noise`` are re-exported from
:mod:`indicate.normalize`: the shipped package needs them for query-time keying,
and this build-only package is excluded from the wheel, so they cannot live here.

``indicate.indic_utils.detect_indic_script`` already classifies text by script,
but it builds a fifteen-entry range table and scans every character against
every script on each call. Frequency mining runs this check tens of millions of
times, so this module keeps a flat range tuple with an early exit. The two agree
on what counts as Indic; this one only answers the yes/no question.
"""

from __future__ import annotations

import unicodedata

from indicate.normalize import EDGE_NOISE, strip_edge_noise

#: Codepoint ranges for the scripts the corpus covers, plus Arabic for Urdu.
INDIC_RANGES = (
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali, Assamese
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0600, 0x06FF),  # Arabic, for Urdu
)


def has_indic_content(text: str | None) -> bool:
    """Report whether ``text`` contains at least one Indic-script character.

    Args:
        text: Any string, or ``None``.

    Returns:
        True if any character falls in :data:`INDIC_RANGES`. Bare codes and
        already-Latin text return False -- they carry no transliteration signal.
    """
    if not text:
        return False
    for char in text:
        code = ord(char)
        for low, high in INDIC_RANGES:
            if low <= code <= high:
                return True
    return False


def is_clean_native_token(token: str | None) -> bool:
    """Report whether ``token`` is usable as a native-side gazetteer key.

    A usable token carries at least one Indic *letter* and no Latin letters.
    Sources glue initials onto names ("A.P.C.<devanagari name>") and sometimes
    emit a bare combining mark as its own token; both are tokenization
    artifacts rather than words anyone would look up.

    Args:
        token: A candidate native-side token.

    Returns:
        True if the token holds an Indic letter and no ASCII letters.
    """
    if not token or any(char.isascii() and char.isalpha() for char in token):
        return False
    # Indic consonants and independent vowels are category Lo; matras, bindus
    # and viramas are Mn/Mc, so a token of marks alone is an alignment artifact.
    return any(
        unicodedata.category(char) == "Lo"
        and any(low <= ord(char) <= high for low, high in INDIC_RANGES)
        for char in token
    )


#: The single script each language's keys must be written in. Wikidata labels a
#: place with whatever scripts it has, so an unscoped pull for Hindi returns
#: Bengali, Telugu and Urdu spellings too -- along with Latin *translations*
#: rather than romanizations (``ఆఫ్`` -> ``of``, ``ஆலயம்`` -> ``temple``), which
#: no length or plausibility check can catch.
LANGUAGE_SCRIPT = {
    "assamese": (0x0980, 0x09FF),
    "bengali": (0x0980, 0x09FF),
    "gujarati": (0x0A80, 0x0AFF),
    "hindi": (0x0900, 0x097F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
    "marathi": (0x0900, 0x097F),
    "odia": (0x0B00, 0x0B7F),
    "punjabi": (0x0A00, 0x0A7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "urdu": (0x0600, 0x06FF),
}


def is_language_script(token: str | None, lang: str) -> bool:
    """Report whether every letter in ``token`` belongs to ``lang``'s script.

    Args:
        token: A candidate native-side token.
        lang: Corpus language name.

    Returns:
        True if the token has at least one letter and all of them fall in that
        language's block. Unknown languages are not gated, so adding a language
        does not silently discard its whole harvest.
    """
    block = LANGUAGE_SCRIPT.get(lang)
    if block is None:
        return True
    if not token:
        return False
    low, high = block
    letters = [char for char in token if unicodedata.category(char) == "Lo"]
    return bool(letters) and all(low <= ord(char) <= high for char in letters)


__all__ = [
    "EDGE_NOISE",
    "INDIC_RANGES",
    "LANGUAGE_SCRIPT",
    "has_indic_content",
    "is_clean_native_token",
    "is_language_script",
    "strip_edge_noise",
]
