"""Transliterate Indic text, choosing how much machinery you pay for."""

from typing import TYPE_CHECKING, Any

from .api import transliterate, transliterate_batch
from .engine import DEFAULT_ENGINE, BackendsUnavailableError
from .engine import KNOWN as BACKENDS
from .indic_utils import detect_indic_script, detect_language_from_script
from .languages import UnknownLanguageError, UnsupportedPairError, supported, supports

if TYPE_CHECKING:
    from .llm_indic import IndicLLMTransliterator

__all__ = [
    "BACKENDS",
    "DEFAULT_ENGINE",
    "BackendsUnavailableError",
    "IndicLLMTransliterator",
    "UnknownLanguageError",
    "UnsupportedPairError",
    "detect_indic_script",
    "detect_language_from_script",
    "supported",
    "supports",
    "transliterate",
    "transliterate_batch",
]


def __getattr__(name: str) -> Any:
    """Resolve ``IndicLLMTransliterator`` on first use (PEP 562).

    Importing it eagerly pulls in litellm, which is 1.3s of process start —
    more than the rest of the package combined, and pure waste for anyone
    using the local models.
    """
    if name == "IndicLLMTransliterator":
        from .llm_indic import IndicLLMTransliterator

        return IndicLLMTransliterator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
