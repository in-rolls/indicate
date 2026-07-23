"""Utilities for Indic language detection and validation."""

from __future__ import annotations


def detect_indic_script(text: str) -> str | None:
    """Auto-detect Indic script from Unicode ranges.

    Args:
        text: Text to analyze.

    Returns:
        Detected script name or None if not Indic.
    """
    if not text:
        return None

    # Unicode ranges for Indic scripts
    scripts = {
        "devanagari": (0x0900, 0x097F),  # Hindi, Marathi, Sanskrit, Nepali
        "bengali": (0x0980, 0x09FF),  # Bengali, Assamese
        "gurmukhi": (0x0A00, 0x0A7F),  # Punjabi
        "gujarati": (0x0A80, 0x0AFF),  # Gujarati
        "odia": (0x0B00, 0x0B7F),  # Odia
        "tamil": (0x0B80, 0x0BFF),  # Tamil
        "telugu": (0x0C00, 0x0C7F),  # Telugu
        "kannada": (0x0C80, 0x0CFF),  # Kannada
        "malayalam": (0x0D00, 0x0D7F),  # Malayalam
        "sinhala": (0x0D80, 0x0DFF),  # Sinhala
        "thai": (0x0E00, 0x0E7F),  # Thai
        "lao": (0x0E80, 0x0EFF),  # Lao
        "tibetan": (0x0F00, 0x0FFF),  # Tibetan
        "myanmar": (0x1000, 0x109F),  # Myanmar
        "arabic": (0x0600, 0x06FF),  # Arabic, Urdu, Persian
    }

    # Count characters in each script
    script_counts = dict.fromkeys(scripts, 0)

    for char in text:
        code_point = ord(char)
        for script, (start, end) in scripts.items():
            if start <= code_point <= end:
                script_counts[script] += 1
                break

    # Return the script with the most characters
    max_script = max(script_counts, key=lambda s: script_counts[s])
    if script_counts[max_script] > 0:
        return max_script

    # Check for Latin script
    latin_count = sum(1 for char in text if ord(char) < 0x0080)
    if latin_count > len(text) * 0.5:
        return "latin"

    return None


def detect_language_from_script(text: str) -> str | None:
    """Detect the most likely language based on script and context.

    Args:
        text: Text to analyze.

    Returns:
        Detected language name or None.
    """
    script = detect_indic_script(text)

    if not script:
        return None

    # Map scripts to most common languages
    script_to_language = {
        "devanagari": "hindi",  # Could also be Marathi, Sanskrit, Nepali
        "bengali": "bengali",  # Could also be Assamese
        "gurmukhi": "punjabi",
        "gujarati": "gujarati",
        "odia": "odia",
        "tamil": "tamil",
        "telugu": "telugu",
        "kannada": "kannada",
        "malayalam": "malayalam",
        "arabic": "urdu",  # In Indian context, likely Urdu
        "latin": "english",
    }

    return script_to_language.get(script)


def is_indic_script(script: str) -> bool:
    """Check if a script is an Indic script.

    Args:
        script: Script name.

    Returns:
        True if script is Indic, False otherwise.
    """
    indic_scripts = {
        "devanagari",
        "bengali",
        "gurmukhi",
        "gujarati",
        "odia",
        "tamil",
        "telugu",
        "kannada",
        "malayalam",
        "sinhala",
        "tibetan",
        "myanmar",
        "arabic",  # Arabic for Urdu in Indian context
    }

    return script in indic_scripts


def validate_indic_language_pair(source: str, target: str) -> bool:
    """Validate that at least one language is Indic.

    Args:
        source: Source language or script.
        target: Target language or script.

    Returns:
        True if valid (at least one is Indic), False otherwise.
    """
    # Language to script mapping
    language_scripts = {
        "hindi": "devanagari",
        "marathi": "devanagari",
        "sanskrit": "devanagari",
        "bengali": "bengali",
        "punjabi": "gurmukhi",
        "gujarati": "gujarati",
        "odia": "odia",
        "tamil": "tamil",
        "telugu": "telugu",
        "kannada": "kannada",
        "malayalam": "malayalam",
        "urdu": "arabic",
        "english": "latin",
    }

    # Get scripts for languages
    source_script = language_scripts.get(source, source)
    target_script = language_scripts.get(target, target)

    # At least one must be Indic
    return is_indic_script(source_script) or is_indic_script(target_script)


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
