"""The public transliteration API: one function, any supported direction.

``indicate.transliterate(text)`` detects the script, picks the language pair, and
runs the default engine chain. Everything else is a keyword::

    transliterate("राजशेखर चिंतालपति")                    # auto-detected Hindi
    transliterate("ਰਵਿ ਸ਼ਰਮਾ", source="punjabi")
    transliterate("नमस्ते", n=3)                          # n-best
    transliterate("मुंबई", engine="model")                # no table
    transliterate("मुंबई", engine=["lookup", "llm"])      # table intercepts the LLM

Text is split on spaces and each word is resolved independently, which is how the
local model was trained and what lets a chain answer different words from
different backends. Reassembly preserves order, so a mixed result reads back in
the order it was given.
"""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING

from .engine import Candidates, build, resolve_words
from .languages import PAIRS, resolve_pair
from .transliterator import DEFAULT_BEAM

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .rerank import Reranker


def _nbest_combine(word_cands: list[Candidates], n: int) -> list[str]:
    """Beam-combine per-word candidate lists into up to ``n`` ranked phrases.

    Args:
        word_cands: Candidates for each word, best first.
        n: How many phrases to return.

    Returns:
        Up to ``n`` phrases, best first.
    """
    combos: list[tuple[str, float]] = [("", 0.0)]
    for candidates in word_cands:
        options = candidates[:n] or [("", 0.0)]
        combos = [
            (phrase + (" " if phrase else "") + text, score + candidate_score)
            for phrase, score in combos
            for text, candidate_score in options
        ]
        combos.sort(key=lambda x: x[1], reverse=True)
        combos = combos[:n]
    return [phrase for phrase, _ in combos][:n]


def _split(text: object) -> list[str]:
    """Split one input into words, tolerating blanks and non-strings."""
    if isinstance(text, str) and text.strip():
        return text.split(" ")
    return []


def transliterate_batch(
    texts: Sequence[str],
    *,
    source: str | None = None,
    target: str = "english",
    engine: Sequence[str] | str | None = None,
    n: int = 1,
    beam: int | None = None,
    reranker: Reranker | None = None,
    llm=None,
    **llm_kwargs,
) -> list[str] | list[list[str]]:
    """Transliterate many texts at once.

    Batching is what makes the local model usable: every word that reaches the
    decoder across every input is decoded in a single pass.

    Args:
        texts: Source-language texts.
        source: Source language; ``None`` detects it from the texts.
        target: Target language.
        engine: Backend names in order, or ``None`` for ``("lookup", "model")``.
        n: Candidates per input. ``1`` returns one string each.
        beam: Beam width override for the model backend.
        reranker: Optional LM re-ranker, applied to model candidates only.
        llm: An existing ``IndicLLMTransliterator`` to reuse.
        **llm_kwargs: Provider settings for the ``llm`` backend.

    Returns:
        One result per input: a ``str`` each when ``n == 1``, else a list of up
        to ``n`` candidates each.

    Raises:
        UnsupportedPairError: If the direction cannot be detected, or no backend in
            the chain supports it.
    """
    texts = list(texts)
    # Cap the number of *non-blank* texts sampled, not the number of positions
    # inspected. Blank entries are supported and come back as "", so slicing
    # texts[:50] meant a batch opening with 50 blanks had nothing to detect
    # from and raised UnsupportedPairError on input it fully supports.
    sample = " ".join(
        islice((t for t in texts if isinstance(t, str) and t.strip()), 50)
    )
    src, tgt = resolve_pair(source, target, sample)
    pair = PAIRS.get((src, tgt))
    if pair is None:
        # No local pair, but an LLM may still cover it; synthesize a config that
        # names the direction without claiming any local files exist.
        from .languages import Pair

        pair = Pair(src, tgt, f"{src}_to_{tgt}", "", "", 0, 0)

    width = beam if beam is not None else DEFAULT_BEAM
    if n > 1:
        width = max(width, n)
    backends = build(
        engine,
        pair,
        beam=width,
        # The reranker reorders a candidate list, which would change n-best
        # ordering; it has always applied to the single-best path only.
        reranker=reranker if n <= 1 else None,
        llm=llm,
        **llm_kwargs,
    )

    per_text = [_split(text) for text in texts]
    flat = [word for words in per_text for word in words]
    resolved = resolve_words(flat, backends)

    results: list = []
    index = 0
    for words in per_text:
        word_cands = resolved[index : index + len(words)]
        index += len(words)
        if n <= 1:
            results.append(" ".join(c[0][0] if c else "" for c in word_cands))
        else:
            results.append([] if not words else _nbest_combine(word_cands, n))
    return results


def transliterate(
    text: str,
    *,
    source: str | None = None,
    target: str = "english",
    engine: Sequence[str] | str | None = None,
    n: int = 1,
    beam: int | None = None,
    reranker: Reranker | None = None,
    llm=None,
    **llm_kwargs,
) -> str | list[str]:
    """Transliterate one text.

    Args:
        text: Source-language text.
        source: Source language; ``None`` detects it from ``text``.
        target: Target language.
        engine: Backend names in order, or ``None`` for ``("lookup", "model")``.
        n: Number of candidates. ``1`` returns a single string.
        beam: Beam width override for the model backend.
        reranker: Optional LM re-ranker.
        llm: An existing ``IndicLLMTransliterator`` to reuse.
        **llm_kwargs: Provider settings for the ``llm`` backend.

    Returns:
        A ``str`` when ``n == 1``; a list of up to ``n`` candidates otherwise.

    Raises:
        TypeError: If ``text`` is ``None``.
        ValueError: If ``text`` is not a string.
        UnsupportedPairError: If the direction is unsupported or undetectable.
    """
    if text is None:
        raise TypeError("Input cannot be None")
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    return transliterate_batch(
        [text],
        source=source,
        target=target,
        engine=engine,
        n=n,
        beam=beam,
        reranker=reranker,
        llm=llm,
        **llm_kwargs,
    )[0]


# UnsupportedPairError is re-exported from indicate/ and defined in
# languages; listing it here too gives Sphinx two targets for one name.
__all__ = ["transliterate", "transliterate_batch"]
