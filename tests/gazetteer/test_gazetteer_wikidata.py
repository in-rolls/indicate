"""Tests for the Wikidata harvester's query scoping and binding conversion."""

import unittest

from gazetteer.harvest_wikidata import (
    ENTITY_CLASSES,
    build_query,
    rows_from_bindings,
)


def _binding(item: str, native: str, en: str) -> dict:
    return {
        "item": {"value": f"http://www.wikidata.org/entity/{item}"},
        "native": {"value": native},
        "en": {"value": en},
    }


class TestQueryScoping(unittest.TestCase):
    """Unscoped Wikidata produces confidently wrong romanizations.

    A 4,476-label unscoped pull yielded ਮਸੀਹ->christ (130,862 roll occurrences),
    ਪਾਲ->paul (49,929) and ਚੱਕ->chuck (45,518): translations and European
    homographs. Scoping to India is not an optimization, it is the correctness
    condition.
    """

    def test_person_query_restricts_to_indian_citizenship(self):
        query = build_query("pa", "person", limit=10, offset=0)
        self.assertIn("wd:Q5", query)  # instance of human
        self.assertIn("wd:Q668", query)  # India

    def test_place_query_restricts_to_places_in_india(self):
        query = build_query("pa", "geo", limit=10, offset=0)
        self.assertIn("P17", query)  # country
        self.assertIn("wd:Q668", query)

    def test_every_entity_class_is_india_scoped(self):
        for entity in ENTITY_CLASSES:
            self.assertIn("wd:Q668", build_query("hi", entity, limit=1, offset=0))

    def test_query_selects_the_requested_language(self):
        self.assertIn('"pa"', build_query("pa", "person", limit=1, offset=0))
        self.assertIn('"hi"', build_query("hi", "person", limit=1, offset=0))

    def test_query_carries_paging(self):
        query = build_query("pa", "person", limit=250, offset=500)
        self.assertIn("LIMIT 250", query)
        self.assertIn("OFFSET 500", query)

    def test_rejects_an_unknown_entity_class(self):
        with self.assertRaises(ValueError):
            build_query("pa", "spaceship", limit=1, offset=0)


class TestRowsFromBindings(unittest.TestCase):
    def test_decomposes_a_full_name_into_particles(self):
        rows = rows_from_bindings(
            [_binding("Q1", "ਮਨਮੋਹਨ ਸਿੰਘ", "Manmohan Singh")], "person"
        )
        self.assertEqual(
            [(r.native, r.latin) for r in rows],
            [("ਮਨਮੋਹਨ", "manmohan"), ("ਸਿੰਘ", "singh")],
        )

    def test_tags_provenance_with_the_qid(self):
        (row, *_) = rows_from_bindings([_binding("Q42", "ਰਾਜ", "Raj")], "person")
        self.assertEqual(row.source, "wikidata")
        self.assertEqual(row.ref, "Q42")

    def test_normalizes_diacritics_in_labels(self):
        (row,) = rows_from_bindings([_binding("Q1", "ਰਾਜ", "Rāj")], "person")
        self.assertEqual(row.latin, "raj")

    def test_skips_labels_whose_token_counts_disagree(self):
        self.assertEqual(
            rows_from_bindings([_binding("Q1", "ਤਰਨ ਤਾਰਨ ਸਾਹਿਬ", "Tarn Taran")], "geo"),
            [],
        )

    def test_skips_a_label_with_no_indic_content(self):
        # An English-only label pair teaches nothing about transliteration.
        self.assertEqual(
            rows_from_bindings([_binding("Q1", "Delhi", "Delhi")], "geo"), []
        )

    def test_carries_the_entity_class_through(self):
        (row,) = rows_from_bindings([_binding("Q1", "ਬਟਾਲਾ", "Batala")], "geo")
        self.assertEqual(row.entity_type, "geo")

    def test_tolerates_incomplete_bindings(self):
        self.assertEqual(rows_from_bindings([{"item": {"value": "x"}}], "geo"), [])


if __name__ == "__main__":
    unittest.main()
