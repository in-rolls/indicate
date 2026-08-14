"""Script detection and text-shape helpers for Indic input.

The language, script and alias tables these functions used to carry are now in
:mod:`indicate.languages`; there had been four overlapping copies. These are the
thin, text-facing wrappers around them.
"""

from __future__ import annotations

from .languages import (
    INDIC_SCRIPTS,
    LANGUAGES,
    detect,
    detect_script,
)


def detect_indic_script(text: str) -> str | None:
    """Auto-detect the dominant script of a string.

    Args:
        text: Text to analyze.

    Returns:
        Script name, ``"latin"`` for mostly-ASCII text, or ``None``.
    """
    return detect_script(text)


def detect_language_from_script(text: str) -> str | None:
    """Guess the language of a string from its script.

    Args:
        text: Text to analyze.

    Returns:
        Language name, or ``None`` if the script maps to none.
    """
    return detect(text)


def is_indic_script(script: str) -> bool:
    """Report whether a script name is an Indic script.

    Args:
        script: Script name.

    Returns:
        True if Indic.
    """
    return script in INDIC_SCRIPTS


def validate_indic_language_pair(source: str, target: str) -> bool:
    """Report whether at least one side of a pair is Indic.

    Args:
        source: Source language or script name.
        target: Target language or script name.

    Returns:
        True if either side is Indic.
    """

    def script_of(name: str) -> str:
        language = LANGUAGES.get(name)
        return language.script if language else name

    return is_indic_script(script_of(source)) or is_indic_script(script_of(target))


def normalize_text_for_transliteration(text: str) -> str:
    """Normalize Indic text for better transliteration.

    Args:
        text: Input text.

    Returns:
        Normalized text.
    """
    # Remove extra whitespace
    text = " ".join(text.split())

    # Common normalizations for Indic text
    replacements = {
        "।": ".",  # Devanagari danda to period
        "॥": ".",  # Devanagari double danda
        "॰": "",  # Devanagari abbreviation sign
        "₹": "Rs.",  # Rupee symbol
        "–": "-",
        "—": "-",  # Em dash to hyphen
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()


def split_mixed_script_text(text: str) -> list[tuple[str, str]]:
    """Split text containing multiple scripts into segments.

    Args:
        text: Text potentially containing multiple scripts.

    Returns:
        List of (text_segment, script) tuples.
    """
    if not text:
        return []

    segments = []
    current_segment = ""
    current_script = None

    for char in text:
        if char.isspace():
            current_segment += char
            continue

        # Detect script of current character
        char_script = None
        code_point = ord(char)

        # Check major script ranges
        if 0x0900 <= code_point <= 0x097F:
            char_script = "devanagari"
        elif 0x0980 <= code_point <= 0x09FF:
            char_script = "bengali"
        elif 0x0A00 <= code_point <= 0x0A7F:
            char_script = "gurmukhi"
        elif 0x0A80 <= code_point <= 0x0AFF:
            char_script = "gujarati"
        elif 0x0B00 <= code_point <= 0x0B7F:
            char_script = "odia"
        elif 0x0B80 <= code_point <= 0x0BFF:
            char_script = "tamil"
        elif 0x0C00 <= code_point <= 0x0C7F:
            char_script = "telugu"
        elif 0x0C80 <= code_point <= 0x0CFF:
            char_script = "kannada"
        elif 0x0D00 <= code_point <= 0x0D7F:
            char_script = "malayalam"
        elif 0x0600 <= code_point <= 0x06FF:
            char_script = "arabic"
        elif code_point < 0x0080:
            char_script = "latin"
        else:
            char_script = "unknown"

        if current_script is None:
            current_script = char_script
            current_segment = char
        elif char_script == current_script or char_script == "unknown":
            current_segment += char
        else:
            # Script changed, save current segment
            if current_segment.strip():
                segments.append((current_segment.strip(), current_script))
            current_segment = char
            current_script = char_script

    # Add the last segment (a non-empty segment means the loop set current_script)
    if current_segment.strip():
        assert current_script is not None  # noqa: S101 - loop invariant narrowing
        segments.append((current_segment.strip(), current_script))

    return segments


def get_language_info(language: str) -> dict:
    """Get detailed information about a language.

    Args:
        language: Language name.

    Returns:
        Dictionary with language information.
    """
    language_info = {
        "hindi": {
            "native_name": "हिन्दी",
            "script": "devanagari",
            "iso_code": "hi",
            "direction": "ltr",
            "speakers_millions": 600,
            "regions": ["India", "Nepal", "Fiji"],
        },
        "tamil": {
            "native_name": "தமிழ்",
            "script": "tamil",
            "iso_code": "ta",
            "direction": "ltr",
            "speakers_millions": 75,
            "regions": ["India", "Sri Lanka", "Singapore", "Malaysia"],
        },
        "telugu": {
            "native_name": "తెలుగు",
            "script": "telugu",
            "iso_code": "te",
            "direction": "ltr",
            "speakers_millions": 95,
            "regions": ["India"],
        },
        "bengali": {
            "native_name": "বাংলা",
            "script": "bengali",
            "iso_code": "bn",
            "direction": "ltr",
            "speakers_millions": 300,
            "regions": ["India", "Bangladesh"],
        },
        "marathi": {
            "native_name": "मराठी",
            "script": "devanagari",
            "iso_code": "mr",
            "direction": "ltr",
            "speakers_millions": 95,
            "regions": ["India"],
        },
        "gujarati": {
            "native_name": "ગુજરાતી",
            "script": "gujarati",
            "iso_code": "gu",
            "direction": "ltr",
            "speakers_millions": 60,
            "regions": ["India"],
        },
        "kannada": {
            "native_name": "ಕನ್ನಡ",
            "script": "kannada",
            "iso_code": "kn",
            "direction": "ltr",
            "speakers_millions": 45,
            "regions": ["India"],
        },
        "malayalam": {
            "native_name": "മലയാളം",
            "script": "malayalam",
            "iso_code": "ml",
            "direction": "ltr",
            "speakers_millions": 35,
            "regions": ["India"],
        },
        "punjabi": {
            "native_name": "ਪੰਜਾਬੀ",
            "script": "gurmukhi",
            "iso_code": "pa",
            "direction": "ltr",
            "speakers_millions": 125,
            "regions": ["India", "Pakistan"],
        },
        "odia": {
            "native_name": "ଓଡ଼ିଆ",
            "script": "odia",
            "iso_code": "or",
            "direction": "ltr",
            "speakers_millions": 35,
            "regions": ["India"],
        },
        "urdu": {
            "native_name": "اردو",
            "script": "arabic",
            "iso_code": "ur",
            "direction": "rtl",
            "speakers_millions": 70,
            "regions": ["India", "Pakistan"],
        },
        "sanskrit": {
            "native_name": "संस्कृतम्",
            "script": "devanagari",
            "iso_code": "sa",
            "direction": "ltr",
            "speakers_millions": 0.025,  # Classical language
            "regions": ["India"],
        },
        "english": {
            "native_name": "English",
            "script": "latin",
            "iso_code": "en",
            "direction": "ltr",
            "speakers_millions": 1500,
            "regions": ["Worldwide"],
        },
    }

    return language_info.get(
        language.lower(),
        {
            "native_name": language,
            "script": "unknown",
            "iso_code": "",
            "direction": "ltr",
            "speakers_millions": 0,
            "regions": [],
        },
    )
