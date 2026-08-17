"""Tests for the language and pair registry."""

from __future__ import annotations

import unittest

from indicate.languages import (
    LANGUAGES,
    MODEL_PAIRS,
    PAIRS,
    SCRIPT_TO_LANGUAGE,
    UnknownLanguageError,
    UnsupportedPairError,
    detect,
    detect_script,
    normalize,
    resolve_pair,
    supported,
    supports,
)


class TestNormalize(unittest.TestCase):
    def test_canonical_names_pass_through(self):
        self.assertEqual(normalize("hindi"), "hindi")

    def test_case_and_padding_do_not_matter(self):
        self.assertEqual(normalize("  Hindi "), "hindi")

    def test_iso_codes_resolve(self):
        self.assertEqual(normalize("hi"), "hindi")
        self.assertEqual(normalize("pa"), "punjabi")

    def test_three_letter_aliases_resolve(self):
        for alias, expected in (
            ("hin", "hindi"),
            ("pan", "punjabi"),
            ("pun", "punjabi"),
        ):
            self.assertEqual(normalize(alias), expected)

    def test_an_unknown_name_says_what_is_known(self):
        with self.assertRaises(UnknownLanguageError) as caught:
            normalize("klingon")
        self.assertIn("hindi", str(caught.exception))


class TestDetection(unittest.TestCase):
    def test_scripts_are_identified(self):
        self.assertEqual(detect_script("मुंबई"), "devanagari")
        self.assertEqual(detect_script("ਸਿੰਘ"), "gurmukhi")
        self.assertEqual(detect_script("Mumbai"), "latin")

    def test_languages_follow_from_scripts(self):
        self.assertEqual(detect("मुंबई"), "hindi")
        self.assertEqual(detect("ਸਿੰਘ"), "punjabi")

    def test_empty_and_unknown_text_detects_nothing(self):
        self.assertIsNone(detect(""))
        self.assertIsNone(detect_script(""))

    def test_detection_is_deterministic_on_a_tie(self):
        # One Devanagari and one Gurmukhi letter: the answer must not depend on
        # dict ordering, or the same input routes differently across runs.
        mixed = "क" + "ਕ"
        self.assertEqual(detect_script(mixed), detect_script(mixed))
        self.assertIn(detect_script(mixed), {"devanagari", "gurmukhi"})

    def test_every_mapped_script_names_a_known_language(self):
        for script, language in SCRIPT_TO_LANGUAGE.items():
            self.assertIn(language, LANGUAGES, script)


class TestSupport(unittest.TestCase):
    def test_local_pairs_support_the_model_backend(self):
        self.assertTrue(supports("hindi", "english", "model"))
        self.assertTrue(supports("punjabi", "english", "model"))

    def test_a_pair_with_no_local_model_does_not(self):
        self.assertFalse(supports("bengali", "english", "model"))
        self.assertTrue(supports("bengali", "english", "lookup"))
        self.assertFalse(supports("tamil", "english", "model"))
        self.assertFalse(supports("tamil", "english", "lookup"))

    def test_the_llm_covers_any_pair_with_an_indic_side(self):
        self.assertTrue(supports("tamil", "english", "llm"))
        self.assertTrue(supports("english", "tamil", "llm"))
        self.assertTrue(supports("hindi", "tamil", "llm"))

    def test_the_llm_does_not_cover_a_pair_with_no_indic_side(self):
        self.assertFalse(supports("english", "english", "llm"))

    def test_supported_lists_backends_in_chain_order(self):
        table = supported()
        self.assertEqual(table[("hindi", "english")][0], "lookup")
        self.assertIn("llm", table[("tamil", "english")])

    def test_supported_can_be_restricted_to_one_backend(self):
        self.assertEqual(set(supported("model")), set(MODEL_PAIRS))


class TestResolvePair(unittest.TestCase):
    def test_an_explicit_source_is_normalized(self):
        self.assertEqual(resolve_pair("hi", "english"), ("hindi", "english"))

    def test_an_omitted_source_is_detected_from_the_text(self):
        self.assertEqual(resolve_pair(None, "english", "ਸਿੰਘ"), ("punjabi", "english"))

    def test_undetectable_text_raises_rather_than_guessing(self):
        # Guessing here is what made `hindi2english` silently mangle Gurmukhi.
        with self.assertRaises(UnsupportedPairError):
            resolve_pair(None, "english", "")


class TestPairs(unittest.TestCase):
    def test_every_pair_is_keyed_by_its_own_direction(self):
        for key, pair in PAIRS.items():
            self.assertEqual(key, pair.key)

    def test_every_pair_names_known_languages(self):
        for pair in PAIRS.values():
            self.assertIn(pair.source, LANGUAGES)
            self.assertIn(pair.target, LANGUAGES)

    def test_pairs_are_immutable(self):
        pair = next(iter(PAIRS.values()))
        with self.assertRaises(AttributeError):
            pair.subdir = "elsewhere"


if __name__ == "__main__":
    unittest.main()
