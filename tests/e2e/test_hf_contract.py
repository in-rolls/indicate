"""Does the model repo actually contain everything the loader will ask for?

Marked ``live`` and deselected by default, so an ordinary ``pytest`` run never
touches the network. Run it with ``-m live``, on a schedule and on release tags.

This exists because the answer was **no**. ``indicate/lookup.py`` resolves
``<pair>/lookup.tsv.gz`` from the pinned ``gojiberries/indicate`` revision;
that path 404s for
both languages while the weights return 200. Every installed user got an empty
string from ``engine=["lookup"]``, silently, and nothing in the suite noticed
because every test ran against a checkout that had the tables locally.

One ``list_repo_files`` call, no downloads. It is cheap and it is the only test
that can see the difference between "works here" and "works anywhere".
"""

from __future__ import annotations

import pytest

from indicate.languages import PAIRS, supports
from indicate.lookup import DOWNLOADABLE, LOOKUP_FILE
from indicate.resources import HF_REPO, HF_REVISION
from indicate.transliterator import DECODER_FILE, ENCODER_FILE

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def repo_files() -> set[str]:
    """Every file in the pinned model repo revision."""
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import (
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        return set(hub.list_repo_files(HF_REPO, revision=HF_REVISION))
    except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
        # The whole point of this test. A deleted repo or a revision that no
        # longer exists means every fresh install is broken; skipping here
        # would let the release path go green on exactly that.
        pytest.fail(f"{HF_REPO}@{HF_REVISION} is gone: {exc}")
    except Exception as exc:  # pragma: no cover - network dependent
        # Only genuine unreachability is environmental.
        pytest.skip(f"could not reach {HF_REPO}: {exc}")


def test_the_pinned_revision_exists(repo_files: set[str]):
    assert repo_files, f"{HF_REPO}@{HF_REVISION} is empty or missing"


@pytest.mark.parametrize(
    "pair",
    [pair for pair in PAIRS.values() if supports(*pair.key, "model")],
    ids=lambda p: p.subdir,
)
def test_every_weight_the_loader_requests_is_present(pair, repo_files: set[str]):
    for rel in (ENCODER_FILE, DECODER_FILE):
        assert f"{pair.subdir}/{rel}" in repo_files, (
            f"{pair.subdir}/{rel} is not in {HF_REPO}@{HF_REVISION}; "
            "engine=['model'] cannot work for a fresh install"
        )


@pytest.mark.parametrize("pair", list(PAIRS.values()), ids=lambda p: p.subdir)
def test_every_lookup_table_the_code_claims_is_present(pair, repo_files: set[str]):
    # DOWNLOADABLE is the code's claim that a table can be fetched. This checks
    # the upload actually happened before a release advertises it.
    if pair.subdir not in DOWNLOADABLE:
        pytest.skip(
            f"{pair.subdir} is not in lookup.DOWNLOADABLE; the package does not "
            "claim its table can be fetched"
        )
    assert f"{pair.subdir}/{LOOKUP_FILE}" in repo_files, (
        f"{pair.subdir}/{LOOKUP_FILE} is not in {HF_REPO}@{HF_REVISION}, but "
        f"lookup.DOWNLOADABLE claims it is. Either upload it and re-tag, or "
        f"remove {pair.subdir!r} from DOWNLOADABLE so the package stops "
        f"advertising a backend it cannot serve."
    )


def test_the_claim_and_the_repo_agree_in_both_directions(repo_files: set[str]):
    # The inverse check: a table sitting in the repo that DOWNLOADABLE does not
    # list is a backend users could have had and did not.
    published = {
        name.split("/")[0] for name in repo_files if name.endswith(LOOKUP_FILE)
    }
    assert published == set(DOWNLOADABLE), (
        f"the repo publishes tables for {sorted(published)} but "
        f"lookup.DOWNLOADABLE says {sorted(DOWNLOADABLE)}"
    )
