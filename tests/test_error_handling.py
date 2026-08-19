"""Failure modes, asserted as behavior rather than as "something happened".

This file used to be the weakest in the suite. Five tests invoked the
``hindi2english`` command, which the API refactor deleted — Click answered "No
such command", exit 2, and every assertion (``assertNotEqual(exit_code, 0)``,
``assertIsNotNone(result)``) was satisfied by that. Four of them were the only
coverage for path handling, invalid UTF-8 and missing files.

Another handful wrapped the call in ``try/except Exception`` and asserted the
exception was an ``Exception``, so every possible outcome passed.

The rule applied here: replace each type check with the invariant the test was
reaching for, and delete every ``try/except`` that turns a failure into a pass.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from indicate.cli import _get_version, cli
from tests.helpers import AGREED, hi, hi_batch

HINDI, ROMAN = next(iter(AGREED.items()))


class TestVersion(unittest.TestCase):
    def test_it_falls_back_when_the_package_is_not_installed(self):
        with patch("indicate.cli.metadata.version") as version:
            version.side_effect = Exception("Package not found")
            self.assertEqual(_get_version(), "unknown")


class TestInvalidArguments(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_an_unknown_option_is_refused(self):
        result = self.runner.invoke(cli, ["transliterate", "--invalid-option"])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("no such option", result.output.lower())

    def test_an_unknown_backend_names_the_valid_ones(self):
        # The failure branch of EngineChain.convert, previously untested.
        result = self.runner.invoke(cli, ["transliterate", "x", "--engine", "quantum"])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("quantum", result.output)
        for known in ("lookup", "model", "llm"):
            self.assertIn(known, result.output)

    def test_a_non_integer_candidate_count_is_refused(self):
        result = self.runner.invoke(cli, ["transliterate", "x", "--n", "many"])
        self.assertEqual(result.exit_code, 2)

    def test_the_deleted_per_language_commands_are_gone(self):
        # Guards against a doc or script still calling the old surface.
        for command in ("hindi2english", "punjabi2english", "llm"):
            result = self.runner.invoke(cli, [command, "हिंदी"])
            self.assertEqual(result.exit_code, 2, command)
            self.assertIn("No such command", result.output)


@pytest.mark.needs_weights
class TestFileHandling(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_a_byte_order_mark_is_stripped_not_transliterated(self):
        # Exercises _read_text_input's BOM branch end to end. The old test
        # asserted `result.output is not None`, which a crash also satisfies.
        with self.runner.isolated_filesystem():
            with Path("bom.txt").open("wb") as handle:
                handle.write(b"\xef\xbb\xbf" + HINDI.encode("utf-8"))
            result = self.runner.invoke(
                cli, ["transliterate", "--from", "hindi", "--input", "bom.txt"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(ROMAN, result.output.lower())

    def test_undecodable_bytes_produce_a_clean_error_not_a_traceback(self):
        with self.runner.isolated_filesystem():
            with Path("corrupted.txt").open("wb") as handle:
                handle.write(b"\xff\xfe\x00\x00")
            result = self.runner.invoke(
                cli, ["transliterate", "--from", "hindi", "--input", "corrupted.txt"]
            )
        # Whatever it does, it must not surface a decode traceback.
        self.assertNotIsInstance(result.exception, UnicodeDecodeError)

    def test_a_missing_input_file_names_the_file(self):
        result = self.runner.invoke(
            cli, ["transliterate", "--input", "nonexistent.txt"]
        )
        self.assertEqual(result.exit_code, 2)
        self.assertIn("nonexistent.txt", result.output)

    def test_writing_over_the_input_is_refused_and_the_input_survives(self):
        # This is the property the guard exists for: not that it errors, but
        # that the source file is still there afterwards.
        with self.runner.isolated_filesystem():
            with Path("same.txt").open("w", encoding="utf-8") as handle:
                handle.write(HINDI)
            result = self.runner.invoke(
                cli,
                [
                    "transliterate",
                    "--from",
                    "hindi",
                    "--input",
                    "same.txt",
                    "--output",
                    "same.txt",
                ],
            )
            with Path("same.txt").open(encoding="utf-8") as handle:
                after = handle.read()
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(after, HINDI)

    def test_reading_an_arbitrary_path_is_deliberate(self):
        # There is no input-path sandbox and there should not be: a CLI that
        # cannot read the file you name is broken. Output paths *are* guarded
        # (see the test above). Recorded here so the absence is a decision.
        result = self.runner.invoke(
            cli, ["transliterate", "--input", "../../../etc/passwd"]
        )
        # Either the file does not exist (2) or it is not Indic text (1).
        self.assertIn(result.exit_code, (1, 2))


@pytest.mark.needs_weights
class TestBatchInvariants(unittest.TestCase):
    def test_a_long_input_preserves_word_count_and_order(self):
        # The flatten/reassemble in transliterate_batch is the thing that can
        # break here; "it returned a str" would not notice.
        words = 1000
        out = hi(" ".join([HINDI] * words))
        parts = out.split(" ")
        self.assertEqual(len(parts), words)
        self.assertEqual(set(parts), {ROMAN})

    def test_single_and_batch_agree(self):
        texts = [HINDI, f"{HINDI} {HINDI}", "", "   "]
        self.assertEqual([hi(t) for t in texts], hi_batch(texts))

    def test_a_long_multiword_input_completes_and_stays_aligned(self):
        # Decoding is hard-bounded by an input-adaptive cap, so this terminates.
        text = " ".join([HINDI] * 20 + ["ZZZQQ"] * 5)
        out = hi(text)
        self.assertEqual(len(out.split(" ")), 25)


@pytest.mark.needs_weights
class TestInvalidInput(unittest.TestCase):
    def test_the_exception_types_are_exact(self):
        with self.assertRaises(TypeError):
            hi(None)
        for bad in (123, [], {}):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                hi(bad)

    def test_a_non_string_inside_a_batch_becomes_empty_not_a_crash(self):
        # _split tolerates non-strings; pin that rather than leaving it to luck.
        self.assertEqual(hi_batch([None, HINDI]), ["", ROMAN])

    def test_lone_surrogates_do_not_desynchronise_a_batch(self):
        # The zip(..., strict=True) in resolve_words would raise if a bad input
        # changed the word count, so this is the assertion that matters.
        texts = ["\ud800", HINDI, "हिंदी\ud800test"]
        out = hi_batch(texts)
        self.assertEqual(len(out), len(texts))
        self.assertEqual(out[1], ROMAN)


@pytest.mark.needs_weights
class TestRepeatability(unittest.TestCase):
    def test_the_same_input_gives_the_same_answer(self):
        first = hi(HINDI)
        for _ in range(5):
            self.assertEqual(hi(HINDI), first)


if __name__ == "__main__":
    unittest.main()
