#!/usr/bin/env python

"""
Test Hindi to English translation
"""

import unittest

import pytest

import indicate


def hi(text, **kwargs):
    """Transliterate Hindi through the unified API."""
    return indicate.transliterate(text, source="hindi", **kwargs)


def hi_batch(texts, **kwargs):
    """Batch-transliterate Hindi through the unified API."""
    return indicate.transliterate_batch(texts, source="hindi", **kwargs)


def pa(text, **kwargs):
    """Transliterate Punjabi through the unified API."""
    return indicate.transliterate(text, source="punjabi", **kwargs)


def pa_batch(texts, **kwargs):
    """Batch-transliterate Punjabi through the unified API."""
    return indicate.transliterate_batch(texts, source="punjabi", **kwargs)


# Shipped default: known words come from the lookup table, the rest are decoded.
# राजशेखर is a table hit ("rajshekhar"); चिंतालपति is a miss and is decoded.
SHIPPED = [("राजशेखर चिंतालपति", "rajshekhar chintalpati"), ("गौरव सूद", "gaurav sood")]

# The same inputs with the table switched off. This is what the eval harnesses
# measure, and pinning it here is what stops a table change from silently
# masking a model regression.
MODEL_ONLY = [("राजशेखर चिंतालपति", "rajshekar chintalpati"), ("गौरव सूद", "gaurav sood")]


@pytest.mark.needs_weights
class TestHindiToEnglish(unittest.TestCase):
    def test_hindi_to_english(self):
        for hindi, english in SHIPPED:
            self.assertEqual(hi(hindi), english)

    def test_model_only_path_is_unchanged(self):
        # Characterizes the shipped v2 (Aksharantar-scaled) model's deterministic
        # beam-search output, independent of the lookup table.
        for hindi, english in MODEL_ONLY:
            self.assertEqual(hi(hindi, engine=("model",)), english)


if __name__ == "__main__":
    unittest.main()
