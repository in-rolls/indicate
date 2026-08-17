"""Tests for the lookup-table builder's candidate-selection rule."""

from __future__ import annotations

import csv
import gzip
import hashlib
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicate.lookup import Lookup
from training.build_lookup import choose, is_implausible, main


class TestPlausibility(unittest.TestCase):
    def test_a_one_letter_answer_to_a_multi_akshara_word_is_debris(self):
        self.assertTrue(is_implausible("ਰਾਜ", "j"))

    def test_a_wildly_long_answer_is_debris(self):
        # The real case: positional alignment paired हैं with a surname.
        self.assertTrue(is_implausible("हैं", "ghana"))

    def test_ordinary_pairs_are_plausible(self):
        for key, value in (("ਸਿੰਘ", "singh"), ("नयना", "naina"), ("ਨਗਰ", "nagar")):
            self.assertFalse(is_implausible(key, value), f"{key} -> {value}")

    def test_a_key_with_no_aksharas_is_never_plausible(self):
        # A bare virama keyed to a word: the native side carries no letter.
        self.assertTrue(is_implausible("੍", "singh"))

    def test_single_akshara_words_may_be_one_letter(self):
        self.assertFalse(is_implausible("ए", "e"))


class TestChoose(unittest.TestCase):
    def test_the_modal_vote_wins(self):
        self.assertEqual(choose("कुमार", Counter({"kumar": 408, "r": 1})), "kumar")

    def test_a_tie_is_left_to_the_model(self):
        # मुंबई really is attested once each as mumbai, mumabi, mumba and
        # mumbaikar. Picking one is guessing; the model gets "mumbai" right.
        tied = Counter({"mumbai": 1, "mumabi": 1, "mumba": 1, "mumbaikar": 1})
        self.assertIsNone(choose("मुंबई", tied))

    def test_a_tie_resolves_when_only_one_candidate_is_plausible(self):
        # Not a disagreement: a real answer next to alignment debris.
        self.assertEqual(choose("ਰਾਜ", Counter({"j": 1, "raaj": 1})), "raaj")

    def test_plausibility_never_overrides_a_decisive_vote(self):
        # आर is how Devanagari writes the letter R, so the one-character answer
        # is correct even though it looks like debris. A vote with a winner is
        # trusted; only ties consult length.
        self.assertEqual(choose("आर", Counter({"r": 5, "ar0": 1})), "r")

    def test_a_tie_of_equally_implausible_candidates_is_dropped(self):
        self.assertIsNone(choose("ਰਾਜ", Counter({"j": 1, "b": 1})))

    def test_the_result_does_not_depend_on_insertion_order(self):
        forward = choose("कुमार", Counter({"kumar": 408, "r": 1}))
        backward = choose("कुमार", Counter({"r": 1, "kumar": 408}))
        self.assertEqual(forward, backward)


class TestExternalCorpus(unittest.TestCase):
    def test_bengali_compiles_without_copying_the_source_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "bengali.csv.gz"
            with gzip.open(corpus, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("bengali", "english"))
                writer.writerow(("বৰুৱা", "barua"))
            output = root / "lookup.tsv.gz"

            status = main(
                [
                    "--lang",
                    "bengali",
                    "--corpus",
                    str(corpus),
                    "--out",
                    str(output),
                ]
            )

            self.assertEqual(status, 0)
            table = Lookup.from_path(output)
            assert table is not None
            self.assertEqual(table.get("বৰুৱা"), "barua")
            self.assertEqual(
                table.meta["source"], "eroll_transliteration/data/bengali.csv.gz"
            )
            self.assertEqual(
                table.meta["source_sha256"],
                hashlib.sha256(corpus.read_bytes()).hexdigest(),
            )

    def test_bengali_names_the_missing_external_input(self):
        self.assertEqual(main(["--lang", "bengali"]), 1)


ROOT = Path(__file__).resolve().parent.parent
EVAL_CLEAN = {
    "punjabi_to_english": "pa",
    "hindi_to_english": "hi",
}


def _eval_clean_tables() -> dict[str, Path]:
    """Return the eval-clean tables that have actually been built."""
    found = {}
    for subdir, code in EVAL_CLEAN.items():
        path = ROOT / "indicate" / "data" / subdir / "lookup.eval-clean.tsv.gz"
        gold = ROOT / "data" / "dakshina" / f"{code}.translit.sampled.test.tsv"
        if path.is_file() and gold.is_file():
            found[code] = path
    return found


@unittest.skipUnless(_eval_clean_tables(), "no eval-clean tables built")
class TestEvalCleanTablesHaveNoLeakage(unittest.TestCase):
    def test_no_dakshina_test_word_survives(self):
        # The whole purpose of --eval-clean: a benchmark run with the table on
        # must not be scoring the table's memory of the test set.
        from indicate.lookup import Lookup, lookup_key

        for code, path in _eval_clean_tables().items():
            table = Lookup.from_path(path)
            assert table is not None
            gold = ROOT / "data" / "dakshina" / f"{code}.translit.sampled.test.tsv"
            words = {
                key
                for line in gold.read_text(encoding="utf-8").splitlines()
                if line.strip()
                if (key := lookup_key(line.split("\t")[0]))
            }
            present = [word for word in words if table.get(word) is not None]
            self.assertEqual(present, [], f"{code}: {len(present)} test words leaked")


if __name__ == "__main__":
    unittest.main()
