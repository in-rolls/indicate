#!/usr/bin/env python

"""
Test Punjabi (Gurmukhi) to English transliteration.
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


@pytest.mark.needs_weights
class TestPunjabiToEnglish(unittest.TestCase):
    def test_punjabi_to_english(self):
        # Characterizes the shipped v2 Punjabi model's deterministic beam-search
        # output on common Gurmukhi names.
        test_inputs = ["ਰਵਿ ਸ਼ਰਮਾ", "ਸਿੰਘ", "ਕੌਰ", "ਗੁਰਪ੍ਰੀਤ"]
        test_outputs = ["ravi sharma", "singh", "kaur", "gurpreet"]
        for punjabi, english in zip(test_inputs, test_outputs, strict=False):
            self.assertEqual(pa(punjabi), english)


if __name__ == "__main__":
    unittest.main()
