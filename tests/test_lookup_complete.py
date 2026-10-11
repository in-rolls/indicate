"""Whole-field lookup must not silently drop an unresolved name component."""

import pytest

from indicate.lookup import Lookup


def test_complete_text_preserves_layout_and_edges():
    table = Lookup({"राम": "ram", "कुमार": "kumar"}, {})
    assert table.get_text(" (राम)\tकुमार,  12\n") == " (ram)\tkumar,  12\n"
    assert table.get_text("राम अज्ञात") is None
    assert table.get_text(" \t") is None
    assert table.get_text("राम ं") is None
    assert table.get_text("राम Kumar") is None


def test_urdu_complete_text_preserves_diacritics_and_arabic_separators():
    table = Lookup({"علی": "ali", "ہُور": "hoor"}, {})
    assert table.get_text("علی، ہُور ۱۲") == "ali، hoor ۱۲"
    assert table.get_text("علی نامعلوم") is None
    assert table.with_overrides({"ہُور": None}).get_text("علی ہُور") is None


def test_review_can_block_a_generic_hit_without_mutating_original():
    table = Lookup({"सिहं": "singh", "राम": "ram"}, {})
    reviewed = table.with_overrides({"सिहं": None, "राम": "raam"})
    assert reviewed.get_text("राम सिहं") is None
    assert reviewed.get("राम") == "raam"
    assert table.get_text("राम सिहं") == "ram singh"


def test_conflicting_normalized_overrides_fail_closed():
    table = Lookup({}, {})
    with pytest.raises(ValueError, match="Conflicting"):
        table.with_overrides({"राम": "ram", "(राम)": None})


@pytest.mark.parametrize("overrides", [{"": "x"}, {"राम": " "}])
def test_invalid_overrides_fail(overrides):
    with pytest.raises(ValueError, match="nonempty"):
        Lookup({}, {}).with_overrides(overrides)


def test_non_latin_override_is_rejected_but_latin_diacritic_is_allowed():
    table = Lookup({}, {})
    with pytest.raises(ValueError, match="Latin-script"):
        table.with_overrides({"राम": "कुमार"})
    assert table.with_overrides({"राम": "rām"}).get_text("राम") == "rām"


@pytest.mark.parametrize("value", ["123", "!!!", "कुमार", "ram ं"])
def test_invalid_stored_hit_or_override_cannot_produce_a_complete_name(value):
    assert Lookup({"राम": value}, {}).get_text("राम") is None
    with pytest.raises(ValueError, match="Latin-script"):
        Lookup({}, {}).with_overrides({"राम": value})
