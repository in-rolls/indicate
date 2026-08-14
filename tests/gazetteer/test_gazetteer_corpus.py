"""Tests for the repo-owned corpus harvesters."""

import json
import tempfile
import unittest
from pathlib import Path

from gazetteer.harvest_corpus import harvest_affidavits, harvest_cricinfo

AFFIDAVIT_HEADER = (
    "name_hindi,name_english,"
    "fathers_or_husbands_name_hindi,fathers_or_husbands_name_english\n"
)


class TestHarvestAffidavits(unittest.TestCase):
    def _write(self, body: str) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "affidavits.csv"
        tmp.write_text(AFFIDAVIT_HEADER + body, encoding="utf-8")
        return tmp

    def test_extracts_particles_from_both_name_columns(self):
        path = self._write("राज कुमार,Raj Kumar,श्याम लाल,Shyam Lal\n")
        pairs = {(r.native, r.latin) for r in harvest_affidavits(path)}
        self.assertIn(("राज", "raj"), pairs)
        self.assertIn(("कुमार", "kumar"), pairs)
        self.assertIn(("श्याम", "shyam"), pairs)
        self.assertIn(("लाल", "lal"), pairs)

    def test_tags_rows_with_the_affidavits_source(self):
        path = self._write("राज,Raj,,\n")
        (row,) = harvest_affidavits(path)
        self.assertEqual(row.source, "affidavits")
        self.assertEqual(row.entity_type, "person")

    def test_skips_rows_whose_token_counts_disagree(self):
        path = self._write("राज कुमार शर्मा,Raj Kumar,,\n")
        self.assertEqual(harvest_affidavits(path), [])

    def test_skips_blank_cells(self):
        path = self._write(",,,\n")
        self.assertEqual(harvest_affidavits(path), [])

    def test_missing_file_yields_nothing(self):
        self.assertEqual(harvest_affidavits(Path("/nonexistent/x.csv")), [])


class TestHarvestCricinfo(unittest.TestCase):
    def _write(self, records: list[dict]) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "players.json"
        tmp.write_text(json.dumps(records), encoding="utf-8")
        return tmp

    def test_extracts_particles_from_player_names(self):
        path = self._write(
            [{"hindi_name": "सचिन तेंदुलकर", "english_name": "Sachin Tendulkar"}]
        )
        pairs = {(r.native, r.latin) for r in harvest_cricinfo(path)}
        self.assertEqual(pairs, {("सचिन", "sachin"), ("तेंदुलकर", "tendulkar")})

    def test_tags_rows_with_the_cricinfo_source(self):
        path = self._write([{"hindi_name": "सचिन", "english_name": "Sachin"}])
        (row,) = harvest_cricinfo(path)
        self.assertEqual(row.source, "cricinfo")

    def test_ignores_records_missing_either_side(self):
        path = self._write([{"hindi_name": "सचिन"}, {"english_name": "Sachin"}, {}])
        self.assertEqual(harvest_cricinfo(path), [])

    def test_missing_file_yields_nothing(self):
        self.assertEqual(harvest_cricinfo(Path("/nonexistent/x.json")), [])


if __name__ == "__main__":
    unittest.main()
