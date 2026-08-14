#!/usr/bin/env python3
"""
Test API functions for the indicate package.
"""

from __future__ import annotations

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
class TestAPI(unittest.TestCase):
    def test_main_import(self):
        """Test that main import works correctly."""
        # Test the main package import
        result = hi("हिंदी")
        self.assertIsInstance(result, str)
        self.assertEqual(result.lower(), "hindi")

    def test_api_function_call(self):
        """Test the top-level indicate.hindi2english function."""
        result = hi("गौरव")
        self.assertIsInstance(result, str)
        self.assertEqual(result.lower(), "gaurav")

    def test_hindi2english_class_direct(self):
        """Test direct class usage."""
        result = hi("राजशेखर")
        self.assertIsInstance(result, str)
        # Lookup-table hit; the model alone gives "rajshekar".
        self.assertEqual(result.lower(), "rajshekhar")
        self.assertEqual(hi("राजशेखर", engine=("model",)).lower(), "rajshekar")

    def test_a_pair_loads_its_model_once(self):
        """The model for a pair is cached, so weights are read once."""
        from indicate.languages import PAIRS
        from indicate.transliterator import model_for

        pair = PAIRS[("hindi", "english")]
        self.assertIs(model_for(pair), model_for(pair))

    def test_model_paths_come_from_the_pair(self):
        """File locations are derived from the registry, not a class name."""
        from indicate.languages import PAIRS
        from indicate.transliterator import model_for

        pair = PAIRS[("hindi", "english")]
        model = model_for(pair)
        self.assertTrue(model.weights_dir.endswith("saved_weights"))

        input_vocab = model.path(pair.input_vocab)
        self.assertIsInstance(input_vocab, str)
        self.assertTrue(input_vocab.endswith("hindi_tokens.json"))

        target_vocab = model.path(pair.target_vocab)
        self.assertIsInstance(target_vocab, str)
        self.assertTrue(target_vocab.endswith("english_tokens.json"))

    def test_multiple_translations(self):
        """Test multiple translations in sequence."""
        # Shipped defaults: table hits where available, decoded otherwise.
        test_cases = [
            ("हिंदी", "hindi"),
            ("गौरव", "gaurav"),
            ("राजशेखर", "rajshekhar"),
            ("चिंतालपति", "chintalpati"),
        ]

        for hindi, expected in test_cases:
            with self.subTest(hindi=hindi):
                result = hi(hindi)
                self.assertEqual(result.lower(), expected)

    def test_consistent_results(self):
        """Test that same input gives consistent results."""
        input_text = "हिंदी"
        result1 = hi(input_text)
        result2 = hi(input_text)
        result3 = hi(input_text)

        # All three ways should give the same result
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)

    def test_api_return_types(self):
        """Test that all API functions return strings."""
        input_text = "हिंदी"

        result1 = hi(input_text)
        result2 = hi(input_text)
        result3 = hi(input_text)

        self.assertIsInstance(result1, str)
        self.assertIsInstance(result2, str)
        self.assertIsInstance(result3, str)


if __name__ == "__main__":
    unittest.main()
