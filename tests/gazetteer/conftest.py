"""Gate the gazetteer suite.

``gazetteer/`` is the corpus builder. It ships in neither the wheel nor the
sdist, and ``--cov=indicate`` does not measure it — so its 154 tests should not
be collected when the question is "does the shipped package work". They still
run by default, because they cover real code that produces the shipped tables;
``make test-pkg`` is what excludes them.
"""

import pytest

pytest.importorskip("gazetteer", reason="gazetteer/ is not present in this checkout")


def pytest_collection_modifyitems(config, items):
    """Mark everything below this directory as a gazetteer test."""
    for item in items:
        item.add_marker(pytest.mark.gazetteer)
