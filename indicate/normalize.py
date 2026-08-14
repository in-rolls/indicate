"""Canonical key normalization for gazetteer lookup.

Gazetteer keys must be produced identically when the corpus is built and when it
is queried, so this module is the single definition of the ladder. Both the
builder under ``gazetteer/`` and the runtime import from here; nothing
re-implements it.

Lookup keys come in two levels, tried in order:

``LEVEL_EXACT``
    The surface form, minus invisible formatting noise. An exact hit means the
    string was literally observed, which keeps the corpus auditable.
``LEVEL_CANONICAL``
    Unicode-canonical: one encoding per grapheme, ASCII digits. Lossless with
    respect to the word's identity, so it can never merge two different words.

There is deliberately **no lossy lookup level**. An earlier design had one that
folded vowel length, nasalization, gemination and the AA matra, on the theory
that these are scribal noise. Ablating each component against the 3,833 Punjab
electoral-roll surfaces with at least 100 occurrences showed otherwise -- every
component merged more distinct words than duplicate spellings:

=====================  =======  =======  =======
component              merges   benign   harmful
=====================  =======  =======  =======
vowel length              73       43       30
diphthong                 62       25       37
drop nasal                83        6       77
drop addak                73       13       60
drop nukta               130        6      124
drop AA matra            198       23      175
=====================  =======  =======  =======

``ਬੁਟਾ``/``buta`` and ``ਬੂਟਾ``/``boota`` are different names; ``ਰਤਨ``/``ratan``
and ``ਰੱਤਨ``/``rattan`` are different names. A lookup key that merges them
returns a confidently wrong romanization, which is worse than a miss.

So folding survives only as :func:`alias_candidate_key`, which *proposes* pairs
of surface forms that might be spelling variants. A proposal becomes an alias
only when source evidence agrees on the romanization -- a data decision, not a
Unicode one. This module therefore never invents a join.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

#: Bump whenever a level's output changes. Corpora record the version they were
#: built with, so a mismatched reader can refuse to use stale keys.
NORMALIZER_VERSION = 2

LEVEL_EXACT = 0
LEVEL_CANONICAL = 1

_LEVELS = (LEVEL_EXACT, LEVEL_CANONICAL)

# Invisible formatting that varies freely between sources and carries no lexical
# information: ZWJ, ZWNJ, ZWSP, soft hyphen, BOM/ZWNBSP.
_ZERO_WIDTH = str.maketrans(dict.fromkeys("‍‌​­﻿", None))

# NFC decomposes most precomposed nukta letters because they are composition
# exclusions (U+0958 QA -> U+0915 U+093C). These three are *not* exclusions, so
# NFC composes them instead. Either way the two spellings of a letter converge,
# but the family ends up represented inconsistently; decomposing these three
# makes the whole family uniform.
# Derived rather than written out: the decomposed form is invisibly different
# from the composed one, so a hand-typed table silently degrades to a no-op.
_NUKTA_HOLDOUTS = {
    ch: unicodedata.normalize("NFD", ch)
    for ch in (
        "ऩ",  # DEVANAGARI LETTER NNNA  -> NA  + NUKTA
        "ऱ",  # DEVANAGARI LETTER RRA   -> RA  + NUKTA
        "ऴ",  # DEVANAGARI LETTER LLLA  -> LLA + NUKTA
    )
}

# NFKC does not touch Indic digits, so the mapping has to be explicit.
_DIGITS = {
    **{chr(0x0966 + i): str(i) for i in range(10)},  # Devanagari
    **{chr(0x0A66 + i): str(i) for i in range(10)},  # Gurmukhi
}

_CANONICAL_MAP = str.maketrans({**_NUKTA_HOLDOUTS, **_DIGITS})

# --- alias proposal only; see the module docstring for why this is not a key ---

_ALIAS_FOLD = str.maketrans(
    {
        # Vowel length and diphthong pairs collapse onto their first member.
        "ी": "ि",
        "ू": "ु",
        "ै": "े",
        "ौ": "ो",
        "ੀ": "ਿ",
        "ੂ": "ੁ",
        "ੈ": "ੇ",
        "ੌ": "ੋ",
        # Marks whose presence varies between transcriptions of the same name.
        "ा": None,  # Devanagari AA matra
        "ਾ": None,  # Gurmukhi AA matra
        "ँ": None,  # Devanagari candrabindu
        "ं": None,  # Devanagari anusvara
        "ਁ": None,  # Gurmukhi adak bindi
        "ਂ": None,  # Gurmukhi bindi
        "ੰ": None,  # Gurmukhi tippi
        "ੱ": None,  # Gurmukhi addak
        "़": None,  # Devanagari nukta
        "਼": None,  # Gurmukhi nukta
    }
)


#: Administrative codes ("022-<name>"), brackets and stray punctuation cling to
#: corpus tokens on both sides of a label; the name itself is what gets keyed.
EDGE_NOISE = (
    "()[]{}<>.,;:!?\"'`/\\|-\u2013\u2014_*# "
    + "0123456789"
    + "".join(chr(0x0966 + i) for i in range(10))  # Devanagari digits
    + "".join(chr(0x0A66 + i) for i in range(10))  # Gurmukhi digits
)


def strip_edge_noise(token: str) -> str:
    """Remove leading and trailing punctuation and digit codes from a token.

    Args:
        token: A raw corpus or query token.

    Returns:
        The token with edge noise removed; may be empty.
    """
    return token.strip(EDGE_NOISE)


def gaz_key(token: str, *, level: int = LEVEL_CANONICAL) -> str:
    """Return the gazetteer lookup key for ``token``.

    Both levels are identity-preserving: two spellings share a key only when
    they are the same string up to encoding.

    Args:
        token: A surface form, in Indic script or Latin.
        level: ``LEVEL_EXACT`` or ``LEVEL_CANONICAL``.

    Returns:
        The normalized key, or ``""`` for empty or whitespace-only input.

    Raises:
        ValueError: If ``level`` is not a defined lookup level. Level 2 was a
            lossy fold in an earlier revision and is rejected explicitly so
            callers cannot silently reacquire it.
    """
    if level not in _LEVELS:
        raise ValueError(
            f"unknown normalization level: {level!r}; "
            f"expected one of {_LEVELS} (the lossy fold is alias_candidate_key)"
        )
    return _gaz_key(token, level)


# Corpus tokens are Zipf-distributed, so a bounded cache turns hundreds of
# millions of calls into a few tens of thousands of real ones. The cache is on
# the private helper because ``lru_cache`` would key positional and keyword
# calls separately.
@lru_cache(maxsize=1 << 17)
def _gaz_key(token: str, level: int) -> str:
    key = token.translate(_ZERO_WIDTH).strip().casefold()
    if not key or level == LEVEL_EXACT:
        return key
    return unicodedata.normalize("NFC", key).translate(_CANONICAL_MAP)


def latin_form(text: str) -> str:
    """Return the canonical Latin form of a romanization candidate.

    Candidates from different sources have to be comparable before they can be
    voted on. This strips diacritics and punctuation but leaves spelling alone,
    so ``rāj`` and ``raj`` agree while ``kumaari`` and ``kumari`` still count as
    a genuine disagreement for adjudication to resolve.

    Args:
        text: A candidate romanization.

    Returns:
        Lowercased ASCII letters, digits, spaces, apostrophes and hyphens, with
        whitespace collapsed. ``""`` when nothing Latin survives, which is the
        signal that the candidate is not a romanization at all.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    kept = [
        ch
        for ch in decomposed
        if not unicodedata.combining(ch)
        and (ch.isascii() and (ch.isalnum() or ch in " '-"))
    ]
    return " ".join("".join(kept).casefold().split())


def alias_candidate_key(token: str) -> str:
    """Return a lossy key grouping surface forms that *might* be variants.

    This exists to generate alias proposals for the corpus builder -- never to
    look anything up. Two tokens sharing this key are candidates for merging;
    whether they actually merge is decided by whether independent sources agree
    on their romanization. See the module docstring for the measured reason.

    Args:
        token: A surface form, in Indic script or Latin.

    Returns:
        The folded proposal key, or ``""`` for empty or whitespace-only input.
    """
    return gaz_key(token, level=LEVEL_CANONICAL).translate(_ALIAS_FOLD)
