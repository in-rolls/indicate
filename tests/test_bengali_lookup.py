"""Bengali is a downloadable lookup direction, not a pretend local model."""

from unittest.mock import patch

import indicate
from indicate.engine import LookupBackend, build
from indicate.languages import PAIRS, supported, supports
from indicate.lookup import Lookup, lookup_key

BN = PAIRS[("bengali", "english")]
BARUA = "বৰুৱা"


def _table() -> Lookup:
    return Lookup({lookup_key(BARUA): "barua"}, {"convention": "roll"})


def test_bengali_is_lookup_only():
    assert supports("bengali", "english", "lookup")
    assert not supports("bengali", "english", "model")
    assert supported()[BN.key] == ("lookup", "llm")


def test_default_chain_does_not_construct_a_model_backend():
    backends = build(None, BN)
    assert len(backends) == 1
    assert isinstance(backends[0], LookupBackend)


def test_assamese_letters_route_through_the_bengali_lookup():
    with patch("indicate.lookup.Lookup.load", return_value=_table()):
        assert indicate.transliterate(BARUA) == "barua"
