"""Tests for frequency mining: which keys are worth looking up."""

import unittest

from gazetteer.frequency import FrequencyTable, coverage_cutpoints, iter_tokens

# Written as escapes: these two spellings of QA are indistinguishable on screen,
# and a hand-typed pair silently degrades into the same bytes.
QA_PRECOMPOSED = "\u0958"
QA_DECOMPOSED = "\u0915\u093c"


class TestFrequencyTable(unittest.TestCase):
    def test_counts_tokens_under_their_canonical_key(self):
        table = FrequencyTable()
        table.add("ਸਿੰਘ", "person")
        table.add("ਸਿੰਘ", "person")
        self.assertEqual(table.count("ਸਿੰਘ"), 2)

    def test_keeps_every_surface_form_seen_for_a_key(self):
        # A key reached by two encodings must remember both spellings so the
        # alias stage can see them.
        table = FrequencyTable()
        table.add(QA_PRECOMPOSED, "geo")
        table.add(QA_DECOMPOSED, "geo")
        (row,) = table.rows()
        self.assertEqual(row.count, 2)
        self.assertEqual(len(row.surfaces), 2)

    def test_entity_hints_are_a_multiset_not_a_label(self):
        # ਸਿੰਘ is overwhelmingly a surname but also occurs in main_town. The
        # collision is recorded, never resolved by fiat.
        table = FrequencyTable()
        for _ in range(10):
            table.add("ਸਿੰਘ", "person")
        table.add("ਸਿੰਘ", "geo")
        (row,) = table.rows()
        self.assertEqual(row.entities, {"person": 10, "geo": 1})

    def test_ignores_empty_and_whitespace_tokens(self):
        table = FrequencyTable()
        table.add("", "person")
        table.add("   ", "person")
        self.assertEqual(table.rows(), [])

    def test_add_accepts_a_multiplier(self):
        table = FrequencyTable()
        table.add("ਕੌਰ", "person", n=57)
        self.assertEqual(table.count("ਕੌਰ"), 57)

    def test_rows_are_ordered_by_descending_count_then_key(self):
        table = FrequencyTable()
        table.add("ਕੌਰ", "person", n=5)
        table.add("ਸਿੰਘ", "person", n=9)
        table.add("ਰਾਮ", "person", n=5)
        self.assertEqual([r.count for r in table.rows()], [9, 5, 5])
        head, *tail = table.rows()
        self.assertEqual(head.key, "ਸਿੰਘ")
        self.assertEqual([r.key for r in tail], sorted(r.key for r in tail))

    def test_total_tokens_reflects_multipliers(self):
        table = FrequencyTable()
        table.add("ਸਿੰਘ", "person", n=3)
        table.add("ਕੌਰ", "person", n=2)
        self.assertEqual(table.total_tokens, 5)

    def test_count_of_an_unseen_key_is_zero(self):
        self.assertEqual(FrequencyTable().count("ਸਿੰਘ"), 0)


class TestIterTokens(unittest.TestCase):
    """Token extraction from raw corpus fields."""

    def test_splits_on_whitespace(self):
        self.assertEqual(
            list(iter_tokens("\u0a24\u0a30\u0a28 \u0a24\u0a3e\u0a30\u0a28")),
            ["\u0a24\u0a30\u0a28", "\u0a24\u0a3e\u0a30\u0a28"],
        )

    def test_strips_administrative_numeric_prefixes(self):
        # ac_name values look like "022-<name>" and the digits are a code, not
        # part of the name. This one appears 197,296 times in 4M rows.
        self.assertEqual(
            list(iter_tokens("022-\u0a16\u0a4b\u0a2e\u0a15\u0a30\u0a28")),
            ["\u0a16\u0a4b\u0a2e\u0a15\u0a30\u0a28"],
        )

    def test_strips_surrounding_brackets(self):
        self.assertEqual(
            list(iter_tokens("(\u0a2c\u0a1f\u0a3e\u0a32\u0a3e)")),
            ["\u0a2c\u0a1f\u0a3e\u0a32\u0a3e"],
        )

    def test_drops_tokens_with_no_indic_letter(self):
        # Bare codes and already-Latin text carry no transliteration signal.
        self.assertEqual(list(iter_tokens("123")), [])
        self.assertEqual(list(iter_tokens("Batala")), [])
        self.assertEqual(list(iter_tokens("-")), [])

    def test_keeps_indic_tokens_mixed_with_latin_ones(self):
        self.assertEqual(
            list(iter_tokens("Batala \u0a2c\u0a1f\u0a3e\u0a32\u0a3e")),
            ["\u0a2c\u0a1f\u0a3e\u0a32\u0a3e"],
        )

    def test_handles_devanagari(self):
        self.assertEqual(
            list(iter_tokens("\u0938\u093f\u0902\u0939")), ["\u0938\u093f\u0902\u0939"]
        )

    def test_empty_and_none_yield_nothing(self):
        self.assertEqual(list(iter_tokens("")), [])
        self.assertEqual(list(iter_tokens(None)), [])


class TestCoverageCutpoints(unittest.TestCase):
    """The whole strategy rests on these numbers, so they get pinned."""

    def test_reports_how_many_types_reach_each_mass_threshold(self):
        # 50 + 30 + 15 + 5 = 100 tokens across 4 types.
        counts = [50, 30, 15, 5]
        cuts = coverage_cutpoints(counts, marks=(0.5, 0.8, 0.95, 1.0))
        self.assertEqual(cuts[0.5], 1)
        self.assertEqual(cuts[0.8], 2)
        self.assertEqual(cuts[0.95], 3)
        self.assertEqual(cuts[1.0], 4)

    def test_handles_an_exact_threshold_hit(self):
        cuts = coverage_cutpoints([50, 50], marks=(0.5,))
        self.assertEqual(cuts[0.5], 1)

    def test_is_independent_of_input_order(self):
        self.assertEqual(
            coverage_cutpoints([5, 50, 15, 30], marks=(0.5, 0.8)),
            coverage_cutpoints([50, 30, 15, 5], marks=(0.5, 0.8)),
        )

    def test_empty_input_yields_zero_everywhere(self):
        self.assertEqual(coverage_cutpoints([], marks=(0.5, 0.99)), {0.5: 0, 0.99: 0})

    def test_uniform_distribution_needs_proportional_types(self):
        # A flat distribution is the worst case for a trunk strategy: no skew,
        # no cheap win. This guards against a bug that manufactures skew.
        cuts = coverage_cutpoints([1] * 100, marks=(0.5,))
        self.assertEqual(cuts[0.5], 50)


if __name__ == "__main__":
    unittest.main()
