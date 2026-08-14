"""Tests for the length-plausibility filter on candidate pairs."""

import unittest

from gazetteer.plausibility import aksharas, is_plausible_pair


class TestAksharas(unittest.TestCase):
    def test_counts_consonants_and_independent_vowels(self):
        self.assertEqual(aksharas("कमल"), 3)

    def test_matras_and_marks_do_not_count(self):
        # सिंह is स + ि + ं + ह: two letters, two marks.
        self.assertEqual(aksharas("सिंह"), 2)

    def test_gurmukhi_is_counted_the_same_way(self):
        self.assertEqual(aksharas("ਸਿੰਘ"), 2)

    def test_latin_and_empty_input_count_as_zero(self):
        self.assertEqual(aksharas("Delhi"), 0)
        self.assertEqual(aksharas(""), 0)


class TestIsPlausiblePair(unittest.TestCase):
    """The band was measured, not guessed -- see the module docstring."""

    def test_accepts_ordinary_transliterations(self):
        self.assertTrue(is_plausible_pair("कुमार", "kumar"))
        self.assertTrue(is_plausible_pair("सिंह", "singh"))
        self.assertTrue(is_plausible_pair("पठानकोट", "pathankot"))

    def test_rejects_an_english_word_aligned_to_a_single_letter(self):
        # Positional alignment fired on coincidentally equal token counts and
        # produced these against a one-letter initial.
        self.assertFalse(is_plausible_pair("के", "administrative"))
        self.assertFalse(is_plausible_pair("के", "adjacent"))

    def test_rejects_a_numeric_code_aligned_to_a_letter(self):
        self.assertFalse(is_plausible_pair("जे", "227j"))
        self.assertFalse(is_plausible_pair("सी", "128c"))

    def test_rejects_a_run_together_compound(self):
        self.assertFalse(is_plausible_pair("भारत", "deindustrialization"))
        self.assertFalse(is_plausible_pair("यूनिवर्सिटी", "deemed-to-be-university"))

    def test_accepts_a_legitimate_single_letter_initial(self):
        # An initial romanized as one letter is correct, just uninformative.
        self.assertTrue(is_plausible_pair("ए", "a"))

    def test_rejects_a_latin_side_far_too_short(self):
        self.assertFalse(is_plausible_pair("अमरकंटक", "a"))

    def test_native_without_aksharas_is_implausible(self):
        self.assertFalse(is_plausible_pair("123", "abc"))

    def test_empty_sides_are_implausible(self):
        self.assertFalse(is_plausible_pair("", "kumar"))
        self.assertFalse(is_plausible_pair("कुमार", ""))

    def test_band_is_configurable(self):
        self.assertTrue(is_plausible_pair("के", "administrative", high=20.0))


if __name__ == "__main__":
    unittest.main()
