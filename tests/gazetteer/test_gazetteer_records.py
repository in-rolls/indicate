"""Tests for the candidate-row schema shared by harvesters and the adjudicator."""

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from gazetteer.records import CandidateRow, aggregate, read_rows, write_rows


def _row(**overrides) -> CandidateRow:
    base = {
        "native": "ਸਿੰਘ",
        "latin": "singh",
        "source": "wikidata",
        "entity_type": "person",
        "weight": 1.0,
        "ref": "Q1",
    }
    return CandidateRow(**{**base, **overrides})


class TestCandidateRow(unittest.TestCase):
    def test_rejects_an_unregistered_source(self):
        # Provenance has to be checkable, so a typo must not become a source.
        with self.assertRaises(ValueError):
            _row(source="wikidataa")

    def test_rejects_empty_native_or_latin(self):
        with self.assertRaises(ValueError):
            _row(native="")
        with self.assertRaises(ValueError):
            _row(latin="")

    def test_rejects_a_native_side_with_no_indic_content(self):
        with self.assertRaises(ValueError):
            _row(native="Delhi", latin="delhi")

    def test_rejects_a_mixed_script_native_token(self):
        # Affidavit rows glue Latin initials onto a Devanagari name
        # ("A.P.C." + the name). Nobody looks that key up.
        with self.assertRaises(ValueError):
            _row(native="A.\u0905\u0936\u094b\u0915", latin="aashok")

    def test_rejects_a_native_token_that_is_only_combining_marks(self):
        # A stray candrabindu aligned to "late" is an alignment artifact, not a
        # word. Indic letters are category Lo; matras and bindus are Mn/Mc.
        with self.assertRaises(ValueError):
            _row(native="\u0901", latin="late")

    def test_rejects_a_latin_side_that_is_not_normalized(self):
        # Candidates must be comparable before they can be voted on; storing a
        # raw "Rāj" alongside a normalized "raj" would split one candidate in two.
        with self.assertRaises(ValueError):
            _row(latin="Rāj")

    def test_accepts_a_normalized_latin_side(self):
        self.assertEqual(_row(latin="raj").latin, "raj")

    def test_rejects_a_weight_outside_the_unit_interval(self):
        with self.assertRaises(ValueError):
            _row(weight=1.5)
        with self.assertRaises(ValueError):
            _row(weight=-0.1)

    def test_is_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            _row().native = "x"  # type: ignore[misc]


class TestRoundTrip(unittest.TestCase):
    def test_rows_survive_a_write_read_cycle(self):
        rows = [_row(), _row(native="ਕੌਰ", latin="kaur", ref="Q2")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wikidata.punjabi.tsv"
            write_rows(path, rows)
            self.assertEqual(read_rows(path), rows)

    def test_writes_a_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.tsv"
            write_rows(path, [_row()])
            first = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(first.split("\t")[0], "native")

    def test_survives_a_tab_inside_a_field(self):
        # A stray tab in a source label must not shift every later column.
        rows = [_row(ref="a\tb")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.tsv"
            write_rows(path, rows)
            self.assertEqual(read_rows(path), rows)

    def test_reading_a_missing_file_yields_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_rows(Path(tmp) / "absent.tsv"), [])

    def test_empty_row_set_still_writes_a_readable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.tsv"
            write_rows(path, [])
            self.assertEqual(read_rows(path), [])


class TestAggregate(unittest.TestCase):
    """Within-source attestation must survive into the weight.

    Collapsing duplicates and keeping max weight made one accidental alignment
    count as much as thousands of consistent ones, which flattened every margin
    and stopped obvious entries reaching high confidence.
    """

    def test_a_repeatedly_attested_pair_outweighs_a_one_off(self):
        rows = [_row(latin="singh") for _ in range(1000)] + [_row(latin="leo")]
        by_latin = {r.latin: r.weight for r in aggregate(rows)}
        self.assertGreater(by_latin["singh"], 0.9)
        self.assertLess(by_latin["leo"], 0.1)

    def test_weights_within_a_source_sum_to_one(self):
        rows = [_row(latin="singh") for _ in range(3)] + [_row(latin="sing")]
        total = sum(r.weight for r in aggregate(rows))
        self.assertAlmostEqual(total, 1.0)

    def test_sources_are_normalized_independently(self):
        rows = [
            _row(source="wikidata", latin="singh"),
            _row(source="wikidata", latin="singh"),
            _row(source="geonames", latin="sing"),
        ]
        result = {(r.source, r.latin): r.weight for r in aggregate(rows)}
        self.assertAlmostEqual(result[("wikidata", "singh")], 1.0)
        self.assertAlmostEqual(result[("geonames", "sing")], 1.0)

    def test_keys_are_normalized_independently(self):
        rows = [_row(latin="singh"), _row(native="\u0a15\u0a4c\u0a30", latin="kaur")]
        for row in aggregate(rows):
            self.assertAlmostEqual(row.weight, 1.0)

    def test_collapses_duplicates_into_one_row(self):
        rows = [_row(latin="singh") for _ in range(5)]
        self.assertEqual(len(aggregate(rows)), 1)

    def test_output_is_deterministic(self):
        rows = [_row(latin="b"), _row(latin="a"), _row(latin="a")]
        first = [(r.latin, r.weight) for r in aggregate(rows)]
        second = [(r.latin, r.weight) for r in aggregate(list(reversed(rows)))]
        self.assertEqual(first, second)

    def test_empty_input_yields_nothing(self):
        self.assertEqual(aggregate([]), [])


if __name__ == "__main__":
    unittest.main()
