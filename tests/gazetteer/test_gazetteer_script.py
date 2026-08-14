"""Tests for the script gates on gazetteer keys."""

from __future__ import annotations

import unittest

from gazetteer.script import (
    has_indic_content,
    is_clean_native_token,
    is_language_script,
)


class TestIsCleanNativeToken(unittest.TestCase):
    def test_an_ordinary_word_is_clean(self):
        self.assertTrue(is_clean_native_token("मुंबई"))

    def test_a_token_with_latin_letters_is_not(self):
        # Sources glue initials onto names: "A.P.C.<devanagari name>".
        self.assertFalse(is_clean_native_token("A.P.C.आथिलिंगम"))

    def test_a_bare_combining_mark_is_not_a_word(self):
        # U+0901 DEVANAGARI SIGN CANDRABINDU, alone: a tokenization artifact.
        self.assertFalse(is_clean_native_token("ँ"))

    def test_empty_input_is_not_clean(self):
        self.assertFalse(is_clean_native_token(""))
        self.assertFalse(is_clean_native_token(None))


class TestIsLanguageScript(unittest.TestCase):
    def test_devanagari_belongs_to_hindi(self):
        self.assertTrue(is_language_script("मुंबई", "hindi"))

    def test_other_indic_scripts_do_not(self):
        # Wikidata labels an entity in every script it has, so an unscoped Hindi
        # pull returns these; they carry translations, not romanizations.
        for token in ("অধিকারী", "ఆఫ్", "ਅੰਮ੍ਰਿਤਸਰ", "ஆலயம்", "بربائی"):
            self.assertFalse(is_language_script(token, "hindi"), token)

    def test_gurmukhi_belongs_to_punjabi(self):
        self.assertTrue(is_language_script("ਸਿੰਘ", "punjabi"))
        self.assertFalse(is_language_script("मुंबई", "punjabi"))

    def test_marathi_shares_devanagari_with_hindi(self):
        self.assertTrue(is_language_script("मुंबई", "marathi"))

    def test_combining_marks_do_not_decide_the_script(self):
        # Matras are Mn/Mc, not Lo; only letters are gated, or a word with a
        # shared mark would be judged on the mark rather than its letters.
        self.assertTrue(is_language_script("हिंदी", "hindi"))

    def test_a_token_with_no_letters_belongs_to_nothing(self):
        self.assertFalse(is_language_script("123", "hindi"))
        self.assertFalse(is_language_script("", "hindi"))

    def test_an_unknown_language_is_not_gated(self):
        # Adding a language must not silently discard its entire harvest.
        self.assertTrue(is_language_script("မြန်မာ", "burmese"))


class TestHasIndicContent(unittest.TestCase):
    def test_latin_and_bare_codes_carry_no_signal(self):
        self.assertFalse(has_indic_content("Mumbai"))
        self.assertFalse(has_indic_content("Q12345"))
        self.assertTrue(has_indic_content("मुंबई"))


if __name__ == "__main__":
    unittest.main()
