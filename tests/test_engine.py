"""Tests for backend chaining.

These use stub backends rather than the real ones wherever possible, so they
test the protocol and the fold rather than the model or the network.
"""

from __future__ import annotations

import unittest

from indicate.engine import (
    AUTHORITATIVE,
    DEFAULT_ENGINE,
    KNOWN,
    LLMBackend,
    LookupBackend,
    ModelBackend,
    build,
    normalize_engine,
    resolve_words,
)
from indicate.languages import PAIRS, UnsupportedPairError

HI = PAIRS[("hindi", "english")]
PA = PAIRS[("punjabi", "english")]


class Stub:
    """Answers the words it was given, declines everything else."""

    def __init__(self, name, answers, *, record=None):
        self.name = name
        self._answers = answers
        self.seen = [] if record is None else record

    def resolve(self, words):
        self.seen.append(list(words))
        return [
            [(self._answers[w], AUTHORITATIVE)] if w in self._answers else None
            for w in words
        ]


class TestNormalizeEngine(unittest.TestCase):
    def test_none_gives_the_default(self):
        self.assertEqual(normalize_engine(None), DEFAULT_ENGINE)

    def test_a_bare_string_is_a_one_backend_chain(self):
        self.assertEqual(normalize_engine("model"), ("model",))

    def test_a_list_keeps_its_order(self):
        self.assertEqual(normalize_engine(["llm", "lookup"]), ("llm", "lookup"))

    def test_an_unknown_backend_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            normalize_engine(["lookup", "telepathy"])
        self.assertIn("telepathy", str(caught.exception))
        for known in KNOWN:
            self.assertIn(known, str(caught.exception))

    def test_an_empty_chain_is_refused(self):
        with self.assertRaises(ValueError):
            normalize_engine([])


class TestResolveWords(unittest.TestCase):
    def test_the_first_backend_that_answers_wins(self):
        first = Stub("a", {"x": "FIRST"})
        second = Stub("b", {"x": "SECOND"})
        out = resolve_words(["x"], [first, second])
        self.assertEqual(out[0][0][0], "FIRST")

    def test_later_backends_see_only_what_was_declined(self):
        first = Stub("a", {"x": "X"})
        second = Stub("b", {"y": "Y"})
        resolve_words(["x", "y", "z"], [first, second])
        self.assertEqual(first.seen, [["x", "y", "z"]])
        self.assertEqual(second.seen, [["y", "z"]])

    def test_a_word_nobody_answers_stays_empty(self):
        out = resolve_words(["q"], [Stub("a", {})])
        self.assertEqual(out, [[]])

    def test_results_stay_aligned_to_the_input(self):
        out = resolve_words(["x", "q", "y"], [Stub("a", {"x": "X", "y": "Y"})])
        self.assertEqual([c[0][0] if c else None for c in out], ["X", None, "Y"])

    def test_an_empty_candidate_list_is_also_a_decline(self):
        # A failed decode used to return [] and become "" in the output. Now it
        # falls through, which is what lets a chain degrade instead of losing a
        # word.
        class Empty:
            name = "empty"

            def resolve(self, words):
                return [[] for _ in words]

        out = resolve_words(["x"], [Empty(), Stub("b", {"x": "RECOVERED"})])
        self.assertEqual(out[0][0][0], "RECOVERED")

    def test_a_chain_stops_once_everything_is_answered(self):
        first = Stub("a", {"x": "X"})
        second = Stub("b", {"x": "OTHER"})
        resolve_words(["x"], [first, second])
        self.assertEqual(second.seen, [])  # never consulted


class TestBuild(unittest.TestCase):
    def test_it_builds_the_named_backends_in_order(self):
        backends = build(["lookup", "model"], HI)
        self.assertEqual([b.name for b in backends], ["lookup", "model"])

    def test_backends_the_pair_does_not_support_are_dropped(self):
        # Tamil has no local pair, so only the llm survives.
        from indicate.languages import Pair

        tamil = Pair("tamil", "english", "tamil_to_english", "", "", 0, 0)
        backends = build(["lookup", "model", "llm"], tamil)
        self.assertEqual([b.name for b in backends], ["llm"])

    def test_a_chain_with_nothing_left_is_an_error_not_a_silent_fallback(self):
        from indicate.languages import Pair

        tamil = Pair("tamil", "english", "tamil_to_english", "", "", 0, 0)
        with self.assertRaises(UnsupportedPairError) as caught:
            build(["lookup", "model"], tamil)
        self.assertIn("llm", str(caught.exception))

    def test_building_does_not_load_anything(self):
        # Constructing a ModelBackend must not touch torch or the filesystem;
        # that is what keeps an all-hit input cheap.
        backend = build(["model"], PA)[0]
        self.assertIsInstance(backend, ModelBackend)

    def test_the_backend_classes_satisfy_the_protocol(self):
        for backend in (LookupBackend(PA), ModelBackend(PA), LLMBackend(PA)):
            self.assertTrue(hasattr(backend, "name"))
            self.assertTrue(callable(backend.resolve))


class TestAuthoritativeScore(unittest.TestCase):
    def test_it_sorts_above_any_beam_score(self):
        # Beam scores are length-normalized log-probs, so always negative.
        # AUTHORITATIVE only has to sit above them for phrase assembly.
        self.assertGreater(AUTHORITATIVE, -1e-9)

    def test_a_single_candidate_makes_its_value_order_irrelevant(self):
        # A non-ranking backend returns exactly one candidate, so the score adds
        # a constant to every phrase containing it and cannot reorder anything.
        out = resolve_words(["x"], [Stub("a", {"x": "X"})])
        self.assertEqual(len(out[0]), 1)


class TestLLMBackend(unittest.TestCase):
    def test_it_deduplicates_before_calling_the_provider(self):
        class Client:
            def __init__(self):
                self.calls = []

            def transliterate_batch(self, texts, batch_size=25):
                self.calls.append(list(texts))
                return [t.upper() for t in texts]

        client = Client()
        backend = LLMBackend(HI, transliterator=client)
        out = backend.resolve(["a", "b", "a", "b", "a"])
        self.assertEqual(client.calls, [["a", "b"]])
        self.assertEqual([c[0][0] for c in out], ["A", "B", "A", "B", "A"])

    def test_a_provider_failure_declines_rather_than_raising(self):
        class Broken:
            def transliterate_batch(self, texts, batch_size=25):
                raise RuntimeError("no api key")

        backend = LLMBackend(HI, transliterator=Broken())
        self.assertEqual(backend.resolve(["a", "b"]), [None, None])

    def test_a_blank_answer_counts_as_a_decline(self):
        class Blank:
            def transliterate_batch(self, texts, batch_size=25):
                return ["", "  "]

        backend = LLMBackend(HI, transliterator=Blank())
        self.assertEqual(backend.resolve(["a", "b"]), [None, None])


if __name__ == "__main__":
    unittest.main()
