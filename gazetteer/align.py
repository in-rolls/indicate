"""Positional alignment of bilingual labels into token pairs.

Every harvester faces the same problem: sources publish *labels* (``ਮਨਮੋਹਨ ਸਿੰਘ``
/ ``Manmohan Singh``), but the gazetteer is keyed on *tokens*. Splitting both
sides and pairing them positionally is what turns a few hundred thousand label
pairs into a name-particle lexicon -- and it is the only way to get ``ਸਿੰਘ`` ->
``singh`` out of Wikidata, whose given-name and family-name items carry almost
no Indic labels.

The alignment is deliberately all-or-nothing. Positional correspondence is only
meaningful when both sides have the same number of tokens; when they do not,
the honest result is no data rather than a guess. This mirrors the run-alignment
already used in ``training/extract_punjabi.py``.
"""

from __future__ import annotations

from gazetteer.script import strip_edge_noise
from indicate.normalize import latin_form


def align_tokens(native: str | None, latin: str | None) -> list[tuple[str, str]]:
    """Pair a native-script label with its Latin label, token by token.

    Args:
        native: A label in an Indic script.
        latin: The same label romanized.

    Returns:
        ``(native_surface, latin_form)`` pairs in order, or ``[]`` when the two
        sides cannot be aligned safely. The native side keeps its spelling --
        only edge punctuation and administrative digit prefixes are removed --
        so the caller decides how to key it; the Latin side is normalized so
        candidates from different sources are comparable.
    """
    if not native or not latin:
        return []

    native_tokens = native.split()
    latin_tokens = latin.split()
    if not native_tokens or len(native_tokens) != len(latin_tokens):
        return []

    pairs: list[tuple[str, str]] = []
    for raw_native, latin_token in zip(native_tokens, latin_tokens, strict=True):
        # latin_form already discards brackets and codes; the native side has to
        # be stripped the same way or the two halves key differently.
        native_token = strip_edge_noise(raw_native)
        romanization = latin_form(latin_token)
        if not native_token or not romanization:
            # A token with no Latin content means the positional correspondence
            # has broken down; the whole label is untrustworthy.
            return []
        pairs.append((native_token, romanization))
    return pairs
