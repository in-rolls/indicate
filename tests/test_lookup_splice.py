"""Tests for routing words through the lookup table before the decoder.

The patch target is ``indicate.utils.batch_candidates`` rather than anything in
``indicate.transliterator``, because the model backend imports it inside the
method: torch must not load when every word hits the table.
"""

import unittest
from unittest.mock import patch

import indicate
from indicate.languages import PAIRS
from indicate.lookup import Lookup

PA = PAIRS[("punjabi", "english")]

# The two most frequent tokens in the Punjab roll; both are in the table.
SINGH = "ਸਿੰਘ"
KAUR = "ਕੌਰ"
# Withheld from v2 training by the leakage filter, and scrambled by the model.
NAGAR = "ਨਗਰ"


def pa(text, **kwargs):
    """Transliterate Punjabi through the unified API."""
    return indicate.transliterate(text, source="punjabi", **kwargs)


def pa_batch(texts, **kwargs):
    """Batch-transliterate Punjabi through the unified API."""
    return indicate.transliterate_batch(texts, source="punjabi", **kwargs)


def _table_available() -> bool:
    return Lookup.load(PA.subdir) is not None


@unittest.skipUnless(_table_available(), "punjabi lookup table not built")
class TestLookupRouting(unittest.TestCase):
    def test_a_known_word_comes_from_the_table(self):
        self.assertEqual(pa(SINGH), "singh")

    def test_the_table_can_be_left_out_of_the_chain(self):
        # The escape hatch the eval harnesses use so benchmarks stay honest.
        with patch("indicate.utils.batch_candidates") as decode:
            decode.return_value = [[("MODEL", 0.0)]]
            self.assertEqual(pa(SINGH, engine=("model",)), "MODEL")

    def test_an_all_hit_input_never_invokes_the_decoder(self):
        # This is what keeps torch out of the process entirely.
        with patch("indicate.utils.batch_candidates") as decode:
            result = pa(f"{SINGH} {KAUR}")
        decode.assert_not_called()
        self.assertEqual(result, "singh kaur")

    def test_a_mixed_string_reassembles_in_order(self):
        with patch("indicate.utils.batch_candidates") as decode:
            decode.return_value = [[("MISS", 0.0)]]
            result = pa(f"{SINGH} ZZZ {KAUR}")
        self.assertEqual(result, "singh MISS kaur")

    def test_only_misses_reach_the_decoder(self):
        with patch("indicate.utils.batch_candidates") as decode:
            decode.return_value = [[("MISS", 0.0)]]
            pa_batch([f"{SINGH} ZZZ"])
        words = decode.call_args[0][0]
        self.assertEqual(words, ["ZZZ"])

    def test_batch_keeps_results_aligned_to_inputs(self):
        with patch("indicate.utils.batch_candidates") as decode:
            decode.return_value = [[("MISS", 0.0)]]
            out = pa_batch([SINGH, "ZZZ", KAUR])
        self.assertEqual(out, ["singh", "MISS", "kaur"])

    def test_the_model_scramble_is_fixed(self):
        # Model-only gives "ganag" for this; the table gives the real word.
        self.assertEqual(pa(NAGAR), "nagar")

    def test_empty_and_blank_inputs_are_unchanged(self):
        self.assertEqual(pa(""), "")
        self.assertEqual(pa("   "), "")

    def test_nbest_uses_the_table_answer_for_hits(self):
        with patch("indicate.utils.batch_candidates") as decode:
            out = pa(SINGH, n=3)
        decode.assert_not_called()
        self.assertEqual(out, ["singh"])

    def test_a_table_only_chain_leaves_misses_empty(self):
        # No fallback in the chain, so a miss has nothing to fall through to.
        # This is the "is my corpus already covered?" probe.
        with patch("indicate.utils.batch_candidates") as decode:
            self.assertEqual(pa(f"{SINGH} ZZZ", engine=("lookup",)), "singh ")
        decode.assert_not_called()

    def test_auto_detection_routes_gurmukhi_without_being_told(self):
        # The old CLI encoded the language in the command name, so Gurmukhi fed
        # to hindi2english produced garbage silently.
        self.assertEqual(indicate.transliterate(SINGH), "singh")


if __name__ == "__main__":
    unittest.main()
