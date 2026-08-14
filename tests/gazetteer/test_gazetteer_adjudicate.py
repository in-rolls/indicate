"""Tests for cross-source adjudication: what the corpus actually asserts."""

import unittest

from gazetteer.adjudicate import adjudicate_all, adjudicate_key
from gazetteer.records import CandidateRow


def _row(source: str, latin: str, native: str = "ਲਾਲ", weight: float = 1.0):
    return CandidateRow(
        native=native,
        latin=latin,
        source=source,
        entity_type="person",
        weight=weight,
    )


class TestRanking(unittest.TestCase):
    def test_orders_candidates_by_score(self):
        result = adjudicate_key(
            [
                _row("wikidata", "lal"),
                _row("geonames", "lal"),
                _row("punjab_roll", "laal"),
            ]
        )
        self.assertEqual([c.latin for c in result.candidates], ["lal", "laal"])

    def test_keeps_runners_up_instead_of_discarding_them(self):
        # "Ranked, with provenance" is the product; a modal vote that throws away
        # alternates cannot express disagreement.
        result = adjudicate_key([_row("wikidata", "lal"), _row("punjab_roll", "laal")])
        self.assertEqual(len(result.candidates), 2)

    def test_two_independent_sources_outrank_one(self):
        pair = adjudicate_key([_row("wikidata", "lal"), _row("geonames", "lal")])
        solo = adjudicate_key([_row("wikidata", "lal")])
        self.assertGreater(pair.candidates[0].score, solo.candidates[0].score)

    def test_records_which_sources_supported_the_winner(self):
        result = adjudicate_key([_row("wikidata", "lal"), _row("geonames", "lal")])
        self.assertEqual(set(result.candidates[0].sources), {"wikidata", "geonames"})

    def test_ties_break_deterministically(self):
        forward = adjudicate_key([_row("wikidata", "aaa"), _row("wikidata", "bbb")])
        reverse = adjudicate_key([_row("wikidata", "bbb"), _row("wikidata", "aaa")])
        self.assertEqual(
            [c.latin for c in forward.candidates], [c.latin for c in reverse.candidates]
        )


class TestConfidence(unittest.TestCase):
    def test_two_independent_human_sources_reach_high(self):
        result = adjudicate_key([_row("wikidata", "lal"), _row("geonames", "lal")])
        self.assertEqual(result.confidence, "high")

    def test_a_single_source_cannot_reach_high(self):
        # Unscoped Wikidata alone asserted ਮਸੀਹ->christ. One source is never
        # enough to confer authority, however reputable.
        self.assertEqual(adjudicate_key([_row("wikidata", "lal")]).confidence, "medium")

    def test_derived_sources_do_not_count_as_two_opinions(self):
        # punjabi_corpus was extracted from punjab_roll and the model was trained
        # on it. Counting them separately would manufacture agreement.
        result = adjudicate_key(
            [_row("punjab_roll", "laal"), _row("punjabi_corpus", "laal")]
        )
        self.assertNotEqual(result.confidence, "high")

    def test_machine_generated_sources_alone_never_reach_high(self):
        result = adjudicate_key([_row("punjab_roll", "laal")])
        self.assertEqual(result.confidence, "low")

    def test_a_human_source_plus_a_machine_source_can_reach_high(self):
        result = adjudicate_key([_row("wikidata", "lal"), _row("punjab_roll", "lal")])
        self.assertEqual(result.confidence, "high")

    def test_a_narrow_margin_blocks_high_confidence(self):
        # Two reputable sources that disagree is a disagreement, not authority.
        result = adjudicate_key([_row("wikidata", "lal"), _row("geonames", "laal")])
        self.assertNotEqual(result.confidence, "high")

    def test_reports_the_margin_between_the_top_two(self):
        result = adjudicate_key([_row("wikidata", "lal"), _row("geonames", "laal")])
        self.assertAlmostEqual(
            result.margin, result.candidates[0].score - result.candidates[1].score
        )

    def test_margin_of_an_uncontested_key_is_its_own_score(self):
        result = adjudicate_key([_row("wikidata", "lal")])
        self.assertAlmostEqual(result.margin, result.candidates[0].score)


class TestKeyAndEntity(unittest.TestCase):
    def test_key_is_the_canonical_form_of_the_native_side(self):
        result = adjudicate_key([_row("wikidata", "qila", native="क़िला")])
        self.assertEqual(result.key, "क़िला")

    def test_accumulates_entity_types_across_sources(self):
        rows = [
            CandidateRow(
                native="ਸਿੰਘ",
                latin="singh",
                source="wikidata",
                entity_type="person",
                weight=1.0,
            ),
            CandidateRow(
                native="ਸਿੰਘ",
                latin="singh",
                source="geonames",
                entity_type="geo",
                weight=1.0,
            ),
        ]
        self.assertEqual(adjudicate_key(rows).entity, {"person": 1, "geo": 1})

    def test_rejects_rows_spanning_more_than_one_key(self):
        with self.assertRaises(ValueError):
            adjudicate_key([_row("wikidata", "lal"), _row("wikidata", "kaur", "ਕੌਰ")])

    def test_rejects_an_empty_row_set(self):
        with self.assertRaises(ValueError):
            adjudicate_key([])


class TestScoreBounds(unittest.TestCase):
    def test_scores_stay_within_the_unit_interval(self):
        rows = [
            _row("wikidata", "lal"),
            _row("geonames", "lal"),
            _row("affidavits", "lal"),
            _row("cricinfo", "lal"),
            _row("punjab_roll", "lal"),
        ]
        for candidate in adjudicate_key(rows).candidates:
            self.assertGreaterEqual(candidate.score, 0.0)
            self.assertLessEqual(candidate.score, 1.0)

    def test_within_source_weight_scales_the_contribution(self):
        strong = adjudicate_key([_row("wikidata", "lal", weight=1.0)])
        weak = adjudicate_key([_row("wikidata", "lal", weight=0.2)])
        self.assertGreater(strong.candidates[0].score, weak.candidates[0].score)


class TestAdjudicateAll(unittest.TestCase):
    def test_groups_rows_by_canonical_key(self):
        rows = [
            _row("wikidata", "lal"),
            _row("geonames", "lal"),
            _row("wikidata", "kaur", native="\u0a15\u0a4c\u0a30"),
        ]
        results = adjudicate_all(rows)
        self.assertEqual(
            {r.key for r in results}, {"\u0a32\u0a3e\u0a32", "\u0a15\u0a4c\u0a30"}
        )

    def test_drops_implausible_pairs_before_adjudicating(self):
        # "\u0915\u0947" -> "administrative" is alignment noise, not a competing
        # candidate; letting it through would create a fake disagreement.
        rows = [
            CandidateRow(
                native="\u0915\u0947",
                latin="k",
                source="affidavits",
                entity_type="person",
                weight=1.0,
            ),
            CandidateRow(
                native="\u0915\u0947",
                latin="administrative",
                source="wikidata",
                entity_type="geo",
                weight=1.0,
            ),
        ]
        (result,) = adjudicate_all(rows)
        self.assertEqual([c.latin for c in result.candidates], ["k"])

    def test_a_key_whose_every_pair_is_implausible_is_dropped(self):
        rows = [
            CandidateRow(
                native="\u092d\u093e\u0930\u0924",
                latin="deindustrialization",
                source="wikidata",
                entity_type="geo",
                weight=1.0,
            )
        ]
        self.assertEqual(adjudicate_all(rows), [])

    def test_results_are_ordered_by_key_for_reproducible_builds(self):
        rows = [
            _row("wikidata", "kaur", native="\u0a15\u0a4c\u0a30"),
            _row("wikidata", "lal"),
        ]
        keys = [r.key for r in adjudicate_all(rows)]
        self.assertEqual(keys, sorted(keys))


if __name__ == "__main__":
    unittest.main()
