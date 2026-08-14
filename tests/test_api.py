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


class StubLLM:
    """The whole interface ``LLMBackend`` uses: one batch method.

    Injected via ``transliterate(llm=...)`` rather than patched, so the test
    exercises the real chain construction and only replaces the network call.
    """

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.seen: list[str] = []

    def transliterate_batch(self, words, batch_size=None):
        """Answer from the canned mapping, recording what was asked.

        Args:
            words: Words to transliterate.
            batch_size: Ignored; present because the backend passes it.

        Returns:
            One answer per word, in order.
        """
        self.seen.extend(words)
        return [self.answers.get(word, "") for word in words]


class TestReverseDirection(unittest.TestCase):
    """English to Indic, which the README advertises and nothing checked.

    ``supported()`` reports these directions as available through the ``llm``
    backend, and the README's feature list claims bidirectionality, but a grep
    for ``source="english"`` across the suite returned nothing before this.
    Only the ``llm`` backend can serve them -- there is no local model in that
    direction -- so the assertion is that the pair resolves, an ``llm`` backend
    is built for it, and its answer comes back.
    """

    def test_english_to_hindi_resolves_and_uses_the_llm_backend(self):
        stub = StubLLM({"hello": "हैलो", "world": "वर्ल्ड"})
        out = indicate.transliterate(
            "hello world", source="english", target="hindi", engine=["llm"], llm=stub
        )
        self.assertEqual(out, "हैलो वर्ल्ड")
        self.assertEqual(stub.seen, ["hello", "world"])

    def test_the_reverse_direction_is_advertised_as_supported(self):
        # supported() and the runtime must agree: advertising a direction that
        # then raises is worse than not advertising it.
        self.assertIn(("english", "hindi"), indicate.supported())
        self.assertEqual(indicate.supported()[("english", "hindi")], ("llm",))

    def test_a_local_engine_is_refused_for_the_reverse_direction(self):
        # There is no English->Hindi local model, so asking for one must name
        # what would work rather than silently falling through to a paid call.
        with pytest.raises(indicate.UnsupportedPairError, match="llm"):
            indicate.transliterate(
                "hello", source="english", target="hindi", engine=["lookup", "model"]
            )


if __name__ == "__main__":
    unittest.main()
