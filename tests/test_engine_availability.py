"""Total failure must be loud; partial failure must stay quiet.

The distinction under test is **declined** versus **unavailable**. A backend
that loaded its asset and had no answer has declined, which is the chain working
as designed and must produce no noise. A backend that could not obtain its asset
at all is unavailable, and if every backend is unavailable the caller would
otherwise receive an empty string that is indistinguishable from a real answer
and exits zero.

That was the shipped behavior: with no network, no weights and no table,
``transliterate()`` returned ``""``.
"""

from __future__ import annotations

import unittest
from unittest import mock

from indicate.engine import (
    AUTHORITATIVE,
    BackendsUnavailableError,
    LookupBackend,
    ModelBackend,
    resolve_words,
)
from indicate.languages import PAIRS

PA = PAIRS[("punjabi", "english")]


class Unavailable:
    """A backend that could not obtain its asset."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.unavailable = False

    def resolve(self, words):
        self.unavailable = True
        return [None] * len(words)


class Declines:
    """A backend that loaded fine and simply has no entry for these words."""

    name = "lookup"

    def __init__(self, answers: dict[str, str] | None = None) -> None:
        self.unavailable = False
        self._answers = answers or {}

    def resolve(self, words):
        return [
            [(self._answers[w], AUTHORITATIVE)] if w in self._answers else None
            for w in words
        ]


class TestEverythingUnavailable(unittest.TestCase):
    def test_it_raises_rather_than_returning_empty(self):
        with self.assertRaises(BackendsUnavailableError):
            resolve_words(["ਸਿੰਘ"], [Unavailable("lookup"), Unavailable("model")])

    def test_the_message_names_every_backend_and_a_fix(self):
        with self.assertRaises(BackendsUnavailableError) as caught:
            resolve_words(["ਸਿੰਘ"], [Unavailable("lookup"), Unavailable("model")])
        message = str(caught.exception)
        self.assertIn("lookup", message)
        self.assertIn("model", message)
        self.assertIn("build_lookup.py", message)
        self.assertIn("HF cache", message)


class TestDeclineStaysSilent(unittest.TestCase):
    def test_a_loaded_backend_with_no_entry_does_not_raise(self):
        # engine=["lookup"] over an uncovered corpus is the advertised
        # "is my corpus already covered?" probe. It must stay quiet.
        self.assertEqual(resolve_words(["ZZZ"], [Declines()]), [[]])

    def test_a_partial_decline_keeps_the_words_it_did_answer(self):
        out = resolve_words(["ਸਿੰਘ", "ZZZ"], [Declines({"ਸਿੰਘ": "singh"})])
        rendered = " ".join(c[0][0] if c else "" for c in out)
        self.assertEqual(rendered, "singh ")  # trailing space is the miss

    def test_an_unavailable_backend_followed_by_a_working_one_is_silent(self):
        out = resolve_words(
            ["ਸਿੰਘ"], [Unavailable("lookup"), Declines({"ਸਿੰਘ": "singh"})]
        )
        self.assertEqual(out[0][0][0], "singh")

    def test_one_answered_word_is_enough_to_stay_silent(self):
        # Not "every word was answered" -- any answer means something ran.
        out = resolve_words(
            ["ਸਿੰਘ", "ZZZ"],
            [Declines({"ਸਿੰਘ": "singh"}), Unavailable("model")],
        )
        self.assertEqual(out[0][0][0], "singh")
        self.assertEqual(out[1], [])


class TestEmptyInput(unittest.TestCase):
    def test_no_words_never_raises(self):
        # An empty input has no answer to fail to produce.
        self.assertEqual(resolve_words([], [Unavailable("lookup")]), [])

    def test_blank_text_through_the_public_api_returns_empty(self):
        import indicate

        self.assertEqual(indicate.transliterate("", source="hindi"), "")
        self.assertEqual(indicate.transliterate("   ", source="hindi"), "")


class TestBackendsReportThemselves(unittest.TestCase):
    def test_a_lookup_backend_with_no_table_is_unavailable(self):
        backend = LookupBackend(PA)
        backend._loaded, backend._table = True, None
        backend.resolve(["ਸਿੰਘ"])
        self.assertTrue(backend.unavailable)

    def test_a_model_backend_that_cannot_load_is_unavailable(self):
        backend = ModelBackend(PA)
        with mock.patch(
            "indicate.transliterator.model_for", side_effect=RuntimeError("no weights")
        ):
            answers = backend.resolve(["ਸਿੰਘ"])
        self.assertEqual(answers, [None])
        self.assertTrue(backend.unavailable)

    def test_a_backend_that_answered_is_not_unavailable(self):
        backend = Declines({"ਸਿੰਘ": "singh"})
        backend.resolve(["ਸਿੰਘ"])
        self.assertFalse(backend.unavailable)


if __name__ == "__main__":
    unittest.main()
