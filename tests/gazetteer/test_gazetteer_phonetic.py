"""Tests for rejecting translations that masquerade as romanizations."""

from __future__ import annotations

import unittest

from gazetteer.phonetic import (
    MIN_SIMILARITY,
    is_phonetic_match,
    mechanical_romanization,
    phonetic_similarity,
)

try:
    import indic_transliteration  # noqa: F401

    HAVE_SANSCRIPT = True
except ImportError:  # pragma: no cover - build-time-only dependency
    HAVE_SANSCRIPT = False

# Real rows from an unscoped Wikidata pull. Each has a perfectly ordinary
# length, so the plausibility filter passes every one of them.
TRANSLATIONS = [
    ("उड़ान", "flight"),
    ("क्षेत्रों", "of"),
    ("कब्र", "forbes"),
    ("आलोचकों", "editors"),
]

# Romanizations that differ from the mechanical form but are genuine, including
# the two conventions the corpus most often disagrees with Dakshina about.
ROMANIZATIONS = [
    ("मुंबई", "mumbai"),
    ("गायत्री", "gayathri"),
    ("गाजीपुर", "ghazipur"),
    ("राजशेखर", "rajshekhar"),
    ("सिंह", "singh"),
    ("दिल्ली", "delhi"),
]


@unittest.skipUnless(HAVE_SANSCRIPT, "indic-transliteration not installed")
class TestMechanicalRomanization(unittest.TestCase):
    def test_devanagari_romanizes_without_diacritics(self):
        self.assertEqual(mechanical_romanization("मुंबई", "hindi"), "mumbai")

    def test_gurmukhi_uses_its_own_scheme(self):
        self.assertTrue(mechanical_romanization("ਸਿੰਘ", "punjabi"))

    def test_an_unsupported_language_yields_nothing(self):
        self.assertEqual(mechanical_romanization("मुंबई", "klingon"), "")


@unittest.skipUnless(HAVE_SANSCRIPT, "indic-transliteration not installed")
class TestPhoneticMatch(unittest.TestCase):
    def test_translations_are_rejected(self):
        for key, latin in TRANSLATIONS:
            self.assertFalse(
                is_phonetic_match(key, latin, "hindi"),
                f"{key} -> {latin} scored "
                f"{phonetic_similarity(key, latin, 'hindi'):.2f}",
            )

    def test_genuine_romanizations_survive(self):
        for key, latin in ROMANIZATIONS:
            self.assertTrue(
                is_phonetic_match(key, latin, "hindi"),
                f"{key} -> {latin} scored "
                f"{phonetic_similarity(key, latin, 'hindi'):.2f}",
            )

    def test_the_threshold_actually_separates_the_two_sets(self):
        # Guards the calibration itself: if a future normalizer change collapses
        # the gap, this fails rather than the filter silently going blind.
        worst_good = min(phonetic_similarity(k, v, "hindi") for k, v in ROMANIZATIONS)
        best_bad = max(phonetic_similarity(k, v, "hindi") for k, v in TRANSLATIONS)
        self.assertGreater(worst_good, best_bad)
        self.assertTrue(best_bad < MIN_SIMILARITY <= worst_good)


class TestUnsupportedLanguagesAreNotFiltered(unittest.TestCase):
    def test_a_language_with_no_scheme_never_rejects(self):
        # Returning 1.0 rather than 0.0 matters: a language we cannot romanize
        # mechanically must pass everything, not discard its whole harvest.
        self.assertEqual(phonetic_similarity("مرحبا", "hello", "urdu"), 1.0)
        self.assertTrue(is_phonetic_match("مرحبا", "hello", "urdu"))


if __name__ == "__main__":
    unittest.main()
