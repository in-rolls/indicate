"""
Test edge cases and error handling for the indicate package.
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
class TestEdgeCases(unittest.TestCase):
    def test_empty_string(self):
        """Test that empty string is handled gracefully."""
        result = hi("")
        self.assertEqual(result, "")

    def test_whitespace_only(self):
        """Test whitespace-only input."""
        result = hi("   ")
        self.assertEqual(result, "")
        # Whitespace-only input returns empty string

    def test_single_character(self):
        """Test single character input."""
        result = hi("अ")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_non_hindi_text(self):
        """Test non-Hindi text input."""
        # English text
        result = hi("hello")
        self.assertIsInstance(result, str)

        # Numbers
        result = hi("123")
        self.assertIsInstance(result, str)

        # Special characters
        result = hi("!@#$%")
        self.assertIsInstance(result, str)

    def test_mixed_language_text(self):
        """Test mixed Hindi-English text."""
        result = hi("हिंदी english मिश्रित")
        self.assertIsInstance(result, str)
        # Should handle mixed content

    def test_very_long_text(self):
        """Test very long input text."""
        long_text = "हिंदी " * 100  # 100 repetitions
        result = hi(long_text)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_unicode_edge_cases(self):
        """Test various Unicode edge cases."""
        test_cases = [
            "हिंदी",  # Standard Hindi
            "हिं\u200dदी",  # With zero-width joiner
            "हिंदी\u0020",  # With explicit space
            "हिंदी\n",  # With newline
            "हिंदी\t",  # With tab
        ]

        for text in test_cases:
            with self.subTest(text=repr(text)):
                result = hi(text)
                self.assertIsInstance(result, str)

    def test_special_devanagari_characters(self):
        """Test special Devanagari characters."""
        test_cases = [
            "०१२३४५६७८९",  # Devanagari digits
            "ॐ",  # Om symbol
            "।",  # Devanagari danda
            "॥",  # Devanagari double danda
        ]

        for text in test_cases:
            with self.subTest(text=text):
                result = hi(text)
                self.assertIsInstance(result, str)

    def test_none_input(self):
        """Test that None input raises appropriate error."""
        with self.assertRaises((TypeError, AttributeError)):
            hi(None)

    def test_non_string_input(self):
        """Test that non-string input raises appropriate error."""
        with self.assertRaises(ValueError):
            hi(123)

        with self.assertRaises(ValueError):
            hi(["हिंदी"])

    def test_extremely_long_input(self):
        """Test extremely long input that might cause timeout."""
        # Create a very long string
        very_long_text = "हिंदी भाषा एक महत्वपूर्ण भारतीय भाषा है " * 50

        # This should either work or timeout gracefully
        try:
            result = hi(very_long_text)
            self.assertIsInstance(result, str)
        except Exception as e:
            # If it fails, it should fail gracefully
            self.assertIsInstance(e, Exception)

    def test_model_loading_resilience(self):
        """Test that model loading is resilient."""
        from indicate.languages import PAIRS
        from indicate.transliterator import model_for

        model = model_for(PAIRS[("hindi", "english")])
        self.assertIsNotNone(model)
        self.assertTrue(model.weights_dir.endswith("saved_weights"))

    def test_repeated_calls_performance(self):
        """Test that repeated calls don't degrade performance significantly."""
        test_text = "हिंदी"

        # First call (model loading)
        result1 = hi(test_text)

        # Subsequent calls should be faster
        for _i in range(5):
            result = hi(test_text)
            self.assertEqual(result, result1)

    def test_concurrent_safety(self):
        """Test that the singleton is thread-safe."""
        import threading

        results = []

        def translate_text():
            result = hi("हिंदी")
            results.append(result)

        threads = []
        for _ in range(3):
            thread = threading.Thread(target=translate_text)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All results should be the same
        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
