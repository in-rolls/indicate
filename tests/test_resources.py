"""The contract with Hugging Face: what is requested, and when.

Two things break silently here and neither is visible from any other test.

**The path join.** ``resolve_data`` asks for ``f"{subdir}/{rel}"``. Rename a
directory in the model repo, or change how the loader spells one, and every
download 404s -- which is exactly how ``lookup.tsv.gz`` came to be missing while
the weights were fine. Pinning the literal string is the only way a unit test
can catch that, because the real check needs the network (see
``tests/e2e/test_hf_contract.py``).

**The local-first rule.** A developer with the file in ``indicate/data/`` must
never hit the network, or a checkout silently depends on connectivity. Asserting
"it returned the right path" does not test that; asserting the download function
was *never called* does.

Nothing here touches the network: ``hf_hub_download`` is imported inside the
function, so patching it at its source intercepts every call site.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from indicate.resources import (
    HF_REPO,
    HF_REVISION,
    local_data_path,
    resolve_data,
    try_resolve_data,
)

ENCODER = "saved_weights/encoder.safetensors"
SUBDIR = "hindi_to_english"


@pytest.fixture
def no_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point ``local_data_path`` at an empty directory.

    Without this the test passes or fails depending on whether the developer
    happens to have downloaded the weights into the source tree.

    Args:
        tmp_path: pytest's per-test temporary directory.
        monkeypatch: pytest's attribute patcher.

    Returns:
        The directory that stands in for ``indicate/data``.
    """
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(
        "indicate.resources.local_data_path",
        lambda subdir, rel: str(root / subdir / rel),
    )
    return root


def test_a_local_file_is_used_and_nothing_is_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    present = tmp_path / "encoder.safetensors"
    present.write_bytes(b"weights")
    monkeypatch.setattr(
        "indicate.resources.local_data_path", lambda subdir, rel: str(present)
    )
    with patch("huggingface_hub.hf_hub_download") as download:
        assert resolve_data(SUBDIR, ENCODER) == str(present)
    download.assert_not_called()


def test_a_missing_file_is_requested_by_its_exact_repo_path(no_local):
    with patch(
        "huggingface_hub.hf_hub_download", return_value="/cache/enc"
    ) as download:
        assert resolve_data(SUBDIR, ENCODER) == "/cache/enc"
    # Positional args and the revision keyword, spelled out. A rename on either
    # side of this join is a 404 that only shows up at runtime.
    download.assert_called_once_with(
        "soodoku/indicate",
        "hindi_to_english/saved_weights/encoder.safetensors",
        revision=HF_REVISION,
    )


def test_the_revision_is_pinned_to_a_tag(no_local):
    # A moving ref means an upstream commit can change what users get without
    # any release here. Tags are immutable; branches are not.
    assert HF_REVISION not in {"main", "master", "HEAD", None}
    assert HF_REPO == "soodoku/indicate"


def test_an_override_repo_and_revision_are_honoured(no_local):
    with patch("huggingface_hub.hf_hub_download", return_value="/cache/x") as download:
        resolve_data(SUBDIR, ENCODER, repo="someone/else", revision="v9.9.9")
    repo, path = download.call_args.args
    assert (repo, path) == ("someone/else", f"{SUBDIR}/{ENCODER}")
    assert download.call_args.kwargs["revision"] == "v9.9.9"


def test_try_resolve_returns_none_rather_than_raising(no_local):
    # Optional assets must disable a feature, not fail a transliteration.
    with patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")):
        assert try_resolve_data(SUBDIR, "lookup.tsv.gz") is None


def test_resolve_propagates_the_failure_its_try_variant_swallows(no_local):
    with (
        patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")),
        pytest.raises(OSError),
    ):
        resolve_data(SUBDIR, ENCODER)


def test_local_paths_land_inside_the_installed_package():
    path = Path(local_data_path(SUBDIR, ENCODER))
    import indicate

    assert path.is_relative_to(Path(indicate.__file__).parent / "data")
    assert path.parts[-3:] == (SUBDIR, "saved_weights", "encoder.safetensors")
