"""Tests for the gazetteer key-normalization ladder.

Indic literals that differ only by an invisible mark are written as escapes so
the intended contrast is readable in the source.
"""

import unittest

from indicate.normalize import (
    LEVEL_CANONICAL,
    LEVEL_EXACT,
    alias_candidate_key,
    gaz_key,
    latin_form,
)

QA_PRECOMPOSED = "क़"  # DEVANAGARI LETTER QA
QA_DECOMPOSED = "क़"  # KA + NUKTA
NNNA_PRECOMPOSED = "ऩ"  # DEVANAGARI LETTER NNNA
NNNA_DECOMPOSED = "ऩ"  # NA + NUKTA
ZA_PRECOMPOSED = "ਜ਼"  # GURMUKHI LETTER ZA
ZA_DECOMPOSED = "ਜ਼"  # JA + NUKTA


class TestLevelExact(unittest.TestCase):
    def test_strips_zero_width_joiner(self):
        self.assertEqual(
            gaz_key("क‍म", level=LEVEL_EXACT), gaz_key("कम", level=LEVEL_EXACT)
        )

    def test_strips_zero_width_non_joiner(self):
        self.assertEqual(
            gaz_key("क‌म", level=LEVEL_EXACT), gaz_key("कम", level=LEVEL_EXACT)
        )

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(gaz_key("  सिंह  ", level=LEVEL_EXACT), "सिंह")

    def test_lowercases_latin_so_mixed_script_input_is_safe(self):
        self.assertEqual(gaz_key("Pathankot", level=LEVEL_EXACT), "pathankot")

    def test_empty_input_yields_empty_key(self):
        self.assertEqual(gaz_key("", level=LEVEL_EXACT), "")
        self.assertEqual(gaz_key("   ", level=LEVEL_EXACT), "")

    def test_does_not_unify_nukta_encodings(self):
        # That is level 1's job; level 0 stays literal so hits are auditable.
        self.assertNotEqual(
            gaz_key(QA_PRECOMPOSED, level=LEVEL_EXACT),
            gaz_key(QA_DECOMPOSED, level=LEVEL_EXACT),
        )


class TestLevelCanonical(unittest.TestCase):
    def test_precomposed_nukta_matches_decomposed_nukta(self):
        self.assertEqual(
            gaz_key(QA_PRECOMPOSED, level=LEVEL_CANONICAL),
            gaz_key(QA_DECOMPOSED, level=LEVEL_CANONICAL),
        )

    def test_unifies_nukta_letters_that_nfc_composes(self):
        # NFC *composes* U+0929 while decomposing its sibling U+0958. Level 1
        # settles on one representative -- uniformly decomposed.
        self.assertEqual(
            gaz_key(NNNA_PRECOMPOSED, level=LEVEL_CANONICAL),
            gaz_key(NNNA_DECOMPOSED, level=LEVEL_CANONICAL),
        )
        self.assertEqual(
            gaz_key(NNNA_PRECOMPOSED, level=LEVEL_CANONICAL), NNNA_DECOMPOSED
        )

    def test_unifies_gurmukhi_nukta(self):
        self.assertEqual(
            gaz_key(ZA_PRECOMPOSED, level=LEVEL_CANONICAL),
            gaz_key(ZA_DECOMPOSED, level=LEVEL_CANONICAL),
        )

    def test_folds_devanagari_digits_to_ascii(self):
        self.assertEqual(gaz_key("१२३", level=LEVEL_CANONICAL), "123")

    def test_folds_gurmukhi_digits_to_ascii(self):
        self.assertEqual(gaz_key("੧੨੩", level=LEVEL_CANONICAL), "123")

    def test_nfkc_would_not_fold_indic_digits(self):
        # Documents why the explicit digit table exists.
        import unicodedata

        self.assertNotEqual(unicodedata.normalize("NFKC", "੧੨੩"), "123")


class TestLookupKeysDoNotFoldOnHeuristics(unittest.TestCase):
    """Lookup keys must not merge spellings on a Unicode heuristic alone.

    Ablating each candidate fold over the 3,833 Punjab-roll surfaces with >=100
    occurrences showed every component conflating genuinely distinct words:
    vowel length 43 benign / 30 harmful, diphthong 25/37, drop-nasal 6/77,
    drop-addak 13/60, drop-nukta 6/124, drop-AA-matra 23/175. Folding is
    therefore a proposal mechanism only -- ``gaz_key`` stays conservative and
    variant merging is decided by source evidence.
    """

    def test_vowel_length_is_a_real_contrast(self):
        # ਬੁਟਾ -> buta (4,281) vs ਬੂਟਾ -> boota (4,450): different words.
        self.assertNotEqual(gaz_key("ਬੁਟਾ"), gaz_key("ਬੂਟਾ"))

    def test_gemination_is_a_real_contrast(self):
        # ਰਤਨ -> ratan (15,667) vs ਰੱਤਨ -> rattan (2,237).
        self.assertNotEqual(gaz_key("ਰਤਨ"), gaz_key("ਰੱਤਨ"))

    def test_nasalization_is_a_real_contrast(self):
        # ਰਾਜ -> raj (107,305) vs ਰਾਂਜ -> ranj (696).
        self.assertNotEqual(gaz_key("ਰਾਜ"), gaz_key("ਰਾਂਜ"))

    def test_aa_matra_is_a_real_contrast(self):
        # ਲਾਲ -> lal (141,643) vs ਲਾਲਾ -> lala (341).
        self.assertNotEqual(gaz_key("ਲਾਲ"), gaz_key("ਲਾਲਾ"))

    def test_nukta_is_a_real_contrast(self):
        # ਦਾਸ -> das (42,148) vs ਦਾਸ਼ -> daash (112).
        self.assertNotEqual(gaz_key("ਦਾਸ"), gaz_key("ਦਾਸ਼"))

    def test_keeps_distinct_consonants_distinct(self):
        self.assertNotEqual(gaz_key("ਰਾਮ"), gaz_key("ਰਾਜ"))
        self.assertNotEqual(gaz_key("राम"), gaz_key("राज"))

    def test_default_level_is_canonical(self):
        self.assertEqual(gaz_key("ਪਠਾਨਕੌਟ"), gaz_key("ਪਠਾਨਕੌਟ", level=LEVEL_CANONICAL))


class TestAliasCandidateKey(unittest.TestCase):
    """The aggressive fold proposes alias pairs for source evidence to confirm."""

    def test_proposes_observed_vowel_length_variant(self):
        self.assertEqual(alias_candidate_key("ਕਪੂਰਥਲਾ"), alias_candidate_key("ਕਪੁਰਥਲਾ"))

    def test_proposes_observed_diphthong_variant(self):
        self.assertEqual(alias_candidate_key("ਪਠਾਨਕੌਟ"), alias_candidate_key("ਪਠਾਨਕੋਟ"))

    def test_proposes_observed_missing_aa_matra(self):
        self.assertEqual(alias_candidate_key("ਤਰਨ"), alias_candidate_key("ਤਾਰਨ"))

    def test_proposes_devanagari_vowel_length_variant(self):
        self.assertEqual(alias_candidate_key("दीना"), alias_candidate_key("दिना"))

    def test_still_separates_distinct_consonants(self):
        self.assertNotEqual(alias_candidate_key("ਰਾਮ"), alias_candidate_key("ਰਾਜ"))

    def test_is_never_finer_than_the_lookup_key(self):
        for a, b in ((QA_PRECOMPOSED, QA_DECOMPOSED), ("  सिंह ", "सिंह")):
            self.assertEqual(gaz_key(a), gaz_key(b))
            self.assertEqual(alias_candidate_key(a), alias_candidate_key(b))


class TestLadderProperties(unittest.TestCase):
    def test_each_level_is_idempotent(self):
        for level in (LEVEL_EXACT, LEVEL_CANONICAL):
            for word in ("ਪਠਾਨਕੌਟ", "क़िला", "सिंह", "Pathankot"):
                once = gaz_key(word, level=level)
                self.assertEqual(once, gaz_key(once, level=level), f"{word} @ {level}")

    def test_alias_candidate_key_is_idempotent(self):
        for word in ("ਪਠਾਨਕੌਟ", "क़िला", "सिंह", "Pathankot"):
            once = alias_candidate_key(word)
            self.assertEqual(once, alias_candidate_key(once), word)

    def test_canonical_never_splits_what_exact_merged(self):
        for a, b in ((QA_PRECOMPOSED, QA_PRECOMPOSED), ("ਪਠਾਨਕੌਟ", "ਪਠਾਨਕੌਟ")):
            if gaz_key(a, level=LEVEL_EXACT) == gaz_key(b, level=LEVEL_EXACT):
                self.assertEqual(
                    gaz_key(a, level=LEVEL_CANONICAL), gaz_key(b, level=LEVEL_CANONICAL)
                )

    def test_rejects_unknown_level(self):
        with self.assertRaises(ValueError):
            gaz_key("सिंह", level=99)

    def test_rejects_the_withdrawn_fold_level(self):
        # Level 2 was a lookup level before the ablation. Callers must not get
        # it back silently.
        with self.assertRaises(ValueError):
            gaz_key("सिंह", level=2)


class TestLatinForm(unittest.TestCase):
    """The Latin side needs its own canonical form before candidates can vote."""

    def test_strips_diacritics(self):
        # The Punjab roll labels contain rāj for what every other source spells raj.
        self.assertEqual(latin_form("r\u0101j"), "raj")
        self.assertEqual(latin_form("\u1e63arm\u0101"), "sarma")

    def test_leaves_spelling_differences_alone(self):
        # kumaari vs kumari is a genuine disagreement, not an encoding artifact.
        self.assertEqual(latin_form("Kumaari"), "kumaari")

    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(latin_form("Singh,"), "singh")
        self.assertEqual(latin_form("(Batala)"), "batala")

    def test_keeps_intra_word_apostrophe_and_hyphen(self):
        self.assertEqual(latin_form("D'Souza"), "d'souza")
        self.assertEqual(latin_form("Tarn-Taran"), "tarn-taran")

    def test_collapses_internal_whitespace(self):
        self.assertEqual(latin_form("  Ram   Kumar "), "ram kumar")

    def test_non_latin_input_yields_empty(self):
        # An empty result is the signal that a candidate is not a romanization.
        self.assertEqual(latin_form("\u0938\u093f\u0902\u0939"), "")
        self.assertEqual(latin_form(""), "")


if __name__ == "__main__":
    unittest.main()
