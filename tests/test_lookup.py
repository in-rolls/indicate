"""Tests for the fast word-level lookup table."""

import gzip
import tempfile
import unittest
from pathlib import Path

from indicate.lookup import Lookup

GURMUKHI_SINGH = "ਸਿੰਘ"
GURMUKHI_KAUR = "ਕੌਰ"
QA_PRECOMPOSED = "क़"
QA_DECOMPOSED = "क़"


def _write_table(
    directory: Path,
    rows: tuple[tuple[str, str], ...] = ((GURMUKHI_SINGH, "singh"),),
    *,
    normalizer: int = 2,
    convention: str = "roll",
    extra: str = "",
) -> Path:
    path = directory / "lookup.tsv.gz"
    header = (
        "#indicate-lookup\t1\n"
        f"#normalizer\t{normalizer}\n"
        "#lang\tpunjabi\n"
        f"#convention\t{convention}\n"
        f"#entries\t{len(rows)}\n"
    )
    body = "".join(f"{k}\t{v}\n" for k, v in rows)
    path.write_bytes(gzip.compress((header + body + extra).encode("utf-8")))
    return path


class TestLoading(unittest.TestCase):
    def test_reads_a_well_formed_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = Lookup.from_path(_write_table(Path(tmp)))
            self.assertIsNotNone(table)
            self.assertEqual(len(table), 1)

    def test_missing_file_disables_rather_than_raises(self):
        self.assertIsNone(Lookup.from_path(Path("/nonexistent/lookup.tsv.gz")))

    def test_normalizer_mismatch_disables_the_table(self):
        # Stale keys are worse than no keys: they would miss silently.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_table(Path(tmp), normalizer=1)
            self.assertIsNone(Lookup.from_path(path))

    def test_corrupt_file_disables_rather_than_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lookup.tsv.gz"
            path.write_bytes(b"this is not gzip")
            self.assertIsNone(Lookup.from_path(path))

    def test_skips_malformed_body_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_table(Path(tmp), extra="no-tab-here\n\n")
            table = Lookup.from_path(path)
            self.assertEqual(len(table), 1)

    def test_exposes_its_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = Lookup.from_path(_write_table(Path(tmp), convention="roll"))
            self.assertEqual(table.convention, "roll")


class TestGet(unittest.TestCase):
    def _table(self, tmp: str, rows=((GURMUKHI_SINGH, "singh"),)) -> Lookup:
        return Lookup.from_path(_write_table(Path(tmp), rows))

    def test_returns_the_romanization(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._table(tmp).get(GURMUKHI_SINGH), "singh")

    def test_miss_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(self._table(tmp).get(GURMUKHI_KAUR))

    def test_normalizes_the_query(self):
        # A caller's surface form need not match the stored key byte for byte.
        with tempfile.TemporaryDirectory() as tmp:
            table = self._table(tmp, ((QA_DECOMPOSED, "qa"),))
            self.assertEqual(table.get(QA_PRECOMPOSED), "qa")

    def test_edge_punctuation_is_ignored_when_matching_and_kept_in_the_answer(self):
        # Two separate jobs. Edge noise must not stop a hit -- that is what
        # keying through strip_edge_noise buys -- but it must survive into the
        # output, or answering from the table silently deletes the caller's
        # text. This asserted the deletion until it was caught in review.
        with tempfile.TemporaryDirectory() as tmp:
            table = self._table(tmp)
            self.assertEqual(table.get(f"({GURMUKHI_SINGH})"), "(singh)")
            self.assertEqual(table.get(f"{GURMUKHI_SINGH},"), "singh,")
            self.assertEqual(table.get(f"022-{GURMUKHI_SINGH}"), "022-singh")
            self.assertEqual(table.get(GURMUKHI_SINGH), "singh")

    def test_empty_query_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = self._table(tmp)
            self.assertIsNone(table.get(""))
            self.assertIsNone(table.get("   "))

    def test_keys_are_stored_normalized(self):
        # Build-time and query-time keying must agree, so a table written with a
        # raw surface form is still findable.
        with tempfile.TemporaryDirectory() as tmp:
            table = self._table(tmp, ((f"  {GURMUKHI_SINGH}  ", "singh"),))
            self.assertEqual(table.get(GURMUKHI_SINGH), "singh")


class TestCaching(unittest.TestCase):
    def test_load_caches_by_subdir(self):
        first = Lookup.load("punjabi_to_english")
        second = Lookup.load("punjabi_to_english")
        self.assertIs(first, second)

    def test_load_of_an_unknown_subdir_is_none(self):
        self.assertIsNone(Lookup.load("no_such_language_to_english"))


if __name__ == "__main__":
    unittest.main()
