"""Tests for the GeoNames harvester.

Fixtures are written to a temp directory in the dump's exact column layout, so
these run with no network access.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gazetteer.harvest_geonames import (
    build_rows,
    load_alternates,
    load_ascii_names,
)

# geonameid, name, asciiname, then 16 more columns to reach the 19 the dump has.
MAIN = [
    ["1261481", "New Delhi", "New Delhi", *[""] * 16],
    ["1275339", "Mumbai", "Mumbai", *[""] * 16],
    ["1253405", "Varanasi", "Varanasi", *[""] * 16],
]

# alternateNameId, geonameid, isolanguage, name, isPreferred, isShort,
# isColloquial, isHistoric, from, to
ALTERNATES = [
    ["1", "1261481", "hi", "नई दिल्ली", "1", "", "", "", "", ""],
    ["2", "1261481", "en", "New Delhi", "1", "", "", "", "", ""],
    ["3", "1275339", "hi", "मुंबई", "1", "", "", "", "", ""],
    ["4", "1275339", "en", "Bombay", "", "", "", "1", "", ""],  # historic
    ["5", "1253405", "hi", "वाराणसी", "1", "", "", "", "", ""],
    ["6", "1253405", "link", "https://example.org/varanasi", "", "", "", "", "", ""],
    ["7", "1253405", "wkdt", "Q1088", "", "", "", "", "", ""],
]


def _write(path: Path, records: list[list[str]]) -> Path:
    path.write_text("".join("\t".join(r) + "\n" for r in records), encoding="utf-8")
    return path


class GeoNamesTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.main = _write(root / "IN.txt", MAIN)
        self.alternates = _write(root / "alt.txt", ALTERNATES)

    def tearDown(self):
        self.tmp.cleanup()


class TestLoading(GeoNamesTestBase):
    def test_ascii_names_key_on_geonameid(self):
        self.assertEqual(load_ascii_names(self.main)["1275339"], "Mumbai")

    def test_historic_names_are_dropped(self):
        # Bombay is a real attestation of something, but not of how Mumbai is
        # romanized today.
        self.assertNotIn("en", load_alternates(self.alternates)["1275339"])

    def test_identifier_pseudo_languages_are_dropped(self):
        varanasi = load_alternates(self.alternates)["1253405"]
        self.assertEqual(set(varanasi), {"hi"})


class TestBuildRows(GeoNamesTestBase):
    def _rows(self):
        return build_rows(
            load_ascii_names(self.main), load_alternates(self.alternates), "hi"
        )

    def test_multi_token_names_decompose_positionally(self):
        # This is the only way to obtain the particle नई -> new.
        pairs = {(r.native, r.latin) for r in self._rows()}
        self.assertIn(("नई", "new"), pairs)
        self.assertIn(("दिल्ली", "delhi"), pairs)

    def test_asciiname_is_the_fallback_when_english_is_missing(self):
        # Mumbai's only English alternate was historic and was dropped, so the
        # pairing must fall back to the main table rather than yield nothing.
        pairs = {(r.native, r.latin) for r in self._rows()}
        self.assertIn(("मुंबई", "mumbai"), pairs)

    def test_every_row_is_tagged_as_a_geonames_place(self):
        for row in self._rows():
            self.assertEqual(row.source, "geonames")
            self.assertEqual(row.entity_type, "geo")

    def test_rows_carry_the_geonameid_for_provenance(self):
        refs = {r.ref for r in self._rows()}
        self.assertTrue(refs <= {"1261481", "1275339", "1253405"})

    def test_a_language_with_no_names_yields_nothing(self):
        self.assertEqual(
            build_rows(
                load_ascii_names(self.main), load_alternates(self.alternates), "ta"
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
