"""Tests for positional alignment of bilingual label pairs."""

import unittest

from gazetteer.align import align_tokens


class TestAlignTokens(unittest.TestCase):
    def test_aligns_equal_length_labels(self):
        self.assertEqual(
            align_tokens("ਮਨਮੋਹਨ ਸਿੰਘ", "Manmohan Singh"),
            [("ਮਨਮੋਹਨ", "manmohan"), ("ਸਿੰਘ", "singh")],
        )

    def test_yields_name_particles_from_a_full_name(self):
        # This is the only way to get ਸਿੰਘ->singh out of Wikidata, which has
        # almost no Indic labels on given-name/family-name items.
        pairs = align_tokens("ਗੁਰਪ੍ਰੀਤ ਕੌਰ", "Gurpreet Kaur")
        self.assertIn(("ਕੌਰ", "kaur"), pairs)

    def test_refuses_to_align_mismatched_token_counts(self):
        self.assertEqual(align_tokens("ਤਰਨ ਤਾਰਨ ਸਾਹਿਬ", "Tarn Taran"), [])
        self.assertEqual(align_tokens("ਸਿੰਘ", "Manmohan Singh"), [])

    def test_tolerates_irregular_whitespace(self):
        self.assertEqual(
            align_tokens("  ਮਨਮੋਹਨ   ਸਿੰਘ ", "Manmohan\tSingh"),
            [("ਮਨਮੋਹਨ", "manmohan"), ("ਸਿੰਘ", "singh")],
        )

    def test_normalizes_diacritics_on_the_latin_side(self):
        self.assertEqual(align_tokens("ਰਾਜ", "Rāj"), [("ਰਾਜ", "raj")])

    def test_strips_punctuation_from_latin_tokens(self):
        self.assertEqual(
            align_tokens("ਬਟਾਲਾ ਸ਼ਹਿਰ", "Batala, City"),
            [("ਬਟਾਲਾ", "batala"), ("ਸ਼ਹਿਰ", "city")],
        )

    def test_rejects_pair_when_a_latin_token_has_no_latin_content(self):
        # "।" normalizes to nothing, so the positional correspondence is broken
        # and the whole label is untrustworthy.
        self.assertEqual(align_tokens("ਬਟਾਲਾ ਸ਼ਹਿਰ", "Batala ।"), [])

    def test_rejects_empty_labels(self):
        self.assertEqual(align_tokens("", "Singh"), [])
        self.assertEqual(align_tokens("ਸਿੰਘ", ""), [])
        self.assertEqual(align_tokens("", ""), [])
        self.assertEqual(align_tokens("   ", "  "), [])

    def test_rejects_none_safely(self):
        self.assertEqual(align_tokens(None, "Singh"), [])
        self.assertEqual(align_tokens("ਸਿੰਘ", None), [])

    def test_strips_edge_punctuation_from_the_native_side_too(self):
        # latin_form already drops brackets, so leaving them on the native side
        # produces a key like "(achal" that no lookup can ever match.
        self.assertEqual(
            align_tokens("(\u0905\u091a\u0932)", "(Achal)"),
            [("\u0905\u091a\u0932", "achal")],
        )

    def test_strips_administrative_digit_prefixes_from_the_native_side(self):
        self.assertEqual(
            align_tokens("022-\u0a16\u0a4b\u0a2e\u0a15\u0a30\u0a28", "Khomkaran"),
            [("\u0a16\u0a4b\u0a2e\u0a15\u0a30\u0a28", "khomkaran")],
        )

    def test_rejects_a_native_token_that_is_only_punctuation(self):
        self.assertEqual(align_tokens("\u0905\u091a\u0932 -", "Achal Kumar"), [])

    def test_native_side_keeps_its_surface_form(self):
        # Keying is the caller's decision; alignment must not pre-normalize away
        # the surface the corpus needs to record.
        pairs = align_tokens("ਪਠਾਨਕੌਟ", "Pathankot")
        self.assertEqual(pairs[0][0], "ਪਠਾਨਕੌਟ")


if __name__ == "__main__":
    unittest.main()
