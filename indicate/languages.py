"""What this install can transliterate, and how it knows.

One registry replaces four overlapping tables that had grown up separately:
``IndicLLMTransliterator.INDIC_LANGUAGES`` plus three dicts inside
``indic_utils`` (script ranges, script-to-language, the Indic-script set). They
agreed by maintenance rather than by construction, which is the kind of
agreement that stops holding.

Two things live here and nothing else:

**Languages** -- name, native spelling, script and ISO code, with the aliases
callers actually type (``hi``, ``hin``, ``pan``).

**Pairs** -- the ``(source, target)`` combinations a local seq2seq model exists
for, and the files that model is made of. Replacing the old class-per-pair
(``HindiToEnglish``, ``PunjabiToEnglish``) with data means adding a language is
a dict entry rather than a module.

Support is per *backend*, not global: ``("tamil", "english")`` is answerable by
an LLM and by nothing else on this machine. :func:`supports` says so, and
:class:`UnsupportedPairError` is raised rather than quietly falling through to a
backend that costs money.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

#: Backends that can appear in an engine chain, in no particular order.
BACKENDS = frozenset({"lookup", "model", "llm"})


@dataclass(frozen=True, slots=True)
class Language:
    """A language this package can name.

    Attributes:
        name: Canonical lowercase name, the key into :data:`LANGUAGES`.
        native: The language's name in its own script.
        script: Writing system, a key of :data:`SCRIPT_RANGES`.
        iso: ISO 639-1 code.
    """

    name: str
    native: str
    script: str
    iso: str


LANGUAGES: dict[str, Language] = {
    lang.name: lang
    for lang in (
        Language("hindi", "हिन्दी", "devanagari", "hi"),
        Language("marathi", "मराठी", "devanagari", "mr"),
        Language("sanskrit", "संस्कृतम्", "devanagari", "sa"),
        Language("bengali", "বাংলা", "bengali", "bn"),
        Language("punjabi", "ਪੰਜਾਬੀ", "gurmukhi", "pa"),
        Language("gujarati", "ગુજરાતી", "gujarati", "gu"),
        Language("odia", "ଓଡ଼ିଆ", "odia", "or"),
        Language("tamil", "தமிழ்", "tamil", "ta"),
        Language("telugu", "తెలుగు", "telugu", "te"),
        Language("kannada", "ಕನ್ನಡ", "kannada", "kn"),
        Language("malayalam", "മലയാളം", "malayalam", "ml"),
        Language("urdu", "اردو", "arabic", "ur"),
        Language("english", "English", "latin", "en"),
    )
}

#: Three-letter and colloquial spellings people actually type.
ALIASES: dict[str, str] = {
    "ben": "bengali",
    "eng": "english",
    "guj": "gujarati",
    "hin": "hindi",
    "kan": "kannada",
    "mal": "malayalam",
    "mar": "marathi",
    "ori": "odia",
    "oriya": "odia",
    "pan": "punjabi",
    "pun": "punjabi",
    "san": "sanskrit",
    "tam": "tamil",
    "tel": "telugu",
    "urd": "urdu",
}

#: Unicode blocks used to identify a script. Wider than :data:`LANGUAGES`,
#: because detection should be able to say "this is Thai" and then decline.
SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "odia": (0x0B00, 0x0B7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
    "sinhala": (0x0D80, 0x0DFF),
    "thai": (0x0E00, 0x0E7F),
    "lao": (0x0E80, 0x0EFF),
    "tibetan": (0x0F00, 0x0FFF),
    "myanmar": (0x1000, 0x109F),
    "arabic": (0x0600, 0x06FF),
}

#: Which language a script most likely encodes. Several are ambiguous --
#: Devanagari carries Hindi, Marathi, Sanskrit and Nepali -- so this is a
#: default for auto-detection, never an assertion. Pass ``source=`` to override.
SCRIPT_TO_LANGUAGE: dict[str, str] = {
    "devanagari": "hindi",
    "bengali": "bengali",
    "gurmukhi": "punjabi",
    "gujarati": "gujarati",
    "odia": "odia",
    "tamil": "tamil",
    "telugu": "telugu",
    "kannada": "kannada",
    "malayalam": "malayalam",
    "arabic": "urdu",  # in an Indian context, Urdu rather than Arabic or Persian
    "latin": "english",
}

INDIC_SCRIPTS = frozenset(SCRIPT_RANGES) - {"latin"}


@dataclass(frozen=True, slots=True)
class Pair:
    """A direction a local seq2seq model exists for.

    Attributes:
        source: Source language name.
        target: Target language name.
        subdir: Directory under ``indicate/data`` holding this pair's
            tokenizers, weights and lookup table.
        input_vocab: Source tokenizer filename.
        target_vocab: Target tokenizer filename.
        max_input: Longest source sequence the weights were trained for.
        max_output: Longest output sequence to decode.
    """

    source: str
    target: str
    subdir: str
    input_vocab: str
    target_vocab: str
    max_input: int
    max_output: int

    @property
    def key(self) -> tuple[str, str]:
        """The ``(source, target)`` tuple this pair is registered under."""
        return (self.source, self.target)


PAIRS: dict[tuple[str, str], Pair] = {
    pair.key: pair
    for pair in (
        Pair(
            source="hindi",
            target="english",
            subdir="hindi_to_english",
            input_vocab="hindi_tokens.json",
            target_vocab="english_tokens.json",
            max_input=47,
            max_output=173,
        ),
        Pair(
            source="punjabi",
            target="english",
            subdir="punjabi_to_english",
            input_vocab="punjabi_tokens.json",
            target_vocab="english_tokens.json",
            # Word-level training data: Gurmukhi/English words are short.
            max_input=32,
            max_output=32,
        ),
    )
}


class UnknownLanguageError(ValueError):
    """A language name that is not in :data:`LANGUAGES` or :data:`ALIASES`."""


class UnsupportedPairError(ValueError):
    """No backend in the requested engine can transliterate this direction."""


def normalize(name: str) -> str:
    """Resolve a language name, ISO code or alias to its canonical name.

    Args:
        name: What the caller typed, e.g. ``"Hindi"``, ``"hi"``, ``"hin"``.

    Returns:
        The canonical lowercase name.

    Raises:
        UnknownLanguageError: If nothing matches.
    """
    key = (name or "").strip().lower()
    if key in LANGUAGES:
        return key
    if key in ALIASES:
        return ALIASES[key]
    for language in LANGUAGES.values():
        if key == language.iso:
            return language.name
    raise UnknownLanguageError(
        f"unknown language {name!r}; known: {', '.join(sorted(LANGUAGES))}"
    )


def detect_script(text: str) -> str | None:
    """Identify the dominant script of a string.

    Args:
        text: Any text.

    Returns:
        The script with the most characters, ``"latin"`` if the text is mostly
        ASCII, or ``None`` when neither applies.
    """
    if not text:
        return None
    counts: dict[str, int] = {}
    for char in text:
        code = ord(char)
        for script, (low, high) in SCRIPT_RANGES.items():
            if low <= code <= high:
                counts[script] = counts.get(script, 0) + 1
                break
    if counts:
        # Ties break on script name so detection is deterministic.
        return min(counts, key=lambda s: (-counts[s], s))
    if sum(1 for char in text if ord(char) < 0x80) > len(text) * 0.5:
        return "latin"
    return None


def detect(text: str) -> str | None:
    """Guess the source language of a string from its script.

    Args:
        text: Any text.

    Returns:
        A canonical language name, or ``None`` if the script is unrecognized or
        maps to no language this package names.
    """
    script = detect_script(text)
    return SCRIPT_TO_LANGUAGE.get(script) if script else None


def _lookup_available(pair: Pair) -> bool:
    """Report whether a lookup table could exist for this pair."""
    from .lookup import SHIPPED

    return pair.subdir in SHIPPED


#: What a backend can actually do on this machine, right now.
READY = "ready"
ON_DEMAND = "downloads on first use"
UNAVAILABLE = "unavailable"


def _worst(states: Iterable[str]) -> str:
    """Return the least-available status among several assets.

    A backend needs every one of its files, so a pair is only as ready as its
    weakest part.

    Args:
        states: Per-asset statuses.

    Returns:
        The least available of them.
    """
    rank = {READY: 0, ON_DEMAND: 1, UNAVAILABLE: 2}
    return max(states, key=lambda state: rank.get(state, 2), default=UNAVAILABLE)


def _asset_status(subdir: str, rel: str, downloadable: bool) -> str:
    """Classify one asset without touching the network.

    Args:
        subdir: Per-language directory.
        rel: Path within it.
        downloadable: Whether the package claims this asset is fetchable.

    Returns:
        One of :data:`READY`, :data:`ON_DEMAND` or :data:`UNAVAILABLE`.
    """
    import os

    from .resources import HF_REPO, HF_REVISION, local_data_path

    if os.path.exists(local_data_path(subdir, rel)):
        return READY
    try:
        from huggingface_hub import try_to_load_from_cache

        if isinstance(
            try_to_load_from_cache(HF_REPO, f"{subdir}/{rel}", revision=HF_REVISION),
            str,
        ):
            return READY
    except Exception as exc:  # pragma: no cover - hub optional at runtime
        # A status probe must never be the thing that breaks a transliteration.
        from .logging import get_logger

        get_logger().debug(f"could not probe the HF cache for {subdir}/{rel}: {exc}")
    return ON_DEMAND if downloadable else UNAVAILABLE


def status(source: str, target: str) -> dict[str, str]:
    """Report what each backend can actually do for a direction, offline.

    :func:`supports` answers "is this direction in scope for this backend",
    which is what drives :func:`~indicate.engine.build`. This answers the
    different and more useful question a user asks: will it work right now.
    Reporting ``lookup`` as available when no table exists and none can be
    fetched is how the CLI came to advertise a backend that silently returned
    empty strings.

    Args:
        source: Canonical source language name.
        target: Canonical target language name.

    Returns:
        Backend name to status, for the backends that support the direction.
    """
    from .lookup import DOWNLOADABLE, LOOKUP_FILE
    from .transliterator import DECODER_FILE, ENCODER_FILE

    out: dict[str, str] = {}
    pair = PAIRS.get((source, target))
    if pair is not None:
        if supports(source, target, "lookup"):
            out["lookup"] = _asset_status(
                pair.subdir, LOOKUP_FILE, pair.subdir in DOWNLOADABLE
            )
        if supports(source, target, "model"):
            # Both weight files, not just the encoder. An interrupted download
            # leaves one of them behind, and reporting `ready` there sends the
            # user to a backend that raises the moment it loads.
            out["model"] = _worst(
                _asset_status(pair.subdir, rel, downloadable=True)
                for rel in (ENCODER_FILE, DECODER_FILE)
            )
    if supports(source, target, "llm"):
        out["llm"] = "needs an API key"
    return out


def supports(source: str, target: str, backend: str) -> bool:
    """Report whether one backend can transliterate one direction.

    Args:
        source: Canonical source language name.
        target: Canonical target language name.
        backend: A member of :data:`BACKENDS`.

    Returns:
        True if that backend covers the direction on this machine.
    """
    if backend == "llm":
        # The LLM needs no local asset; it only needs the pair to be Indic on at
        # least one side, which is what the prompt is written for.
        known = source in LANGUAGES and target in LANGUAGES
        indic = LANGUAGES[source].script in INDIC_SCRIPTS if source in LANGUAGES else 0
        indic_t = (
            LANGUAGES[target].script in INDIC_SCRIPTS if target in LANGUAGES else 0
        )
        return bool(known and (indic or indic_t) and source != target)
    pair = PAIRS.get((source, target))
    if pair is None:
        return False
    if backend == "model":
        return True
    if backend == "lookup":
        return _lookup_available(pair)
    return False


def supported(backend: str | None = None) -> dict[tuple[str, str], tuple[str, ...]]:
    """Report every direction this install can transliterate, and by what.

    Args:
        backend: Restrict to one backend, or ``None`` for all of them.

    Returns:
        ``(source, target)`` to the backends that cover it, in chain order.
    """
    wanted = (backend,) if backend else ("lookup", "model", "llm")
    directions: set[tuple[str, str]] = set(PAIRS)
    if "llm" in wanted:
        directions |= {
            (source, target)
            for source in LANGUAGES
            for target in LANGUAGES
            if supports(source, target, "llm")
        }
    out: dict[tuple[str, str], tuple[str, ...]] = {}
    for direction in sorted(directions):
        covered = tuple(b for b in wanted if supports(*direction, b))
        if covered:
            out[direction] = covered
    return out


def resolve_pair(source: str | None, target: str, text: str = "") -> tuple[str, str]:
    """Normalize a requested direction, detecting the source if it was omitted.

    Args:
        source: Source language, or ``None`` to detect from ``text``.
        target: Target language.
        text: Sample text used only when ``source`` is ``None``.

    Returns:
        The canonical ``(source, target)``.

    Raises:
        UnsupportedPairError: If ``source`` is ``None`` and detection fails. Guessing
            would silently transliterate Gurmukhi with a Devanagari model.
    """
    if source is None:
        detected = detect(text)
        if detected is None:
            raise UnsupportedPairError(
                "could not detect the source language from the text; "
                "pass source= explicitly"
            )
        source = detected
    return normalize(source), normalize(target)
