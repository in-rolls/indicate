"""Locate packaged data files, local-first then Hugging Face.

Model weights are too large for the wheel and the lookup tables derive from
corpora with their own license and provenance constraints, so both are resolved
the same way: use the file in ``indicate/data/`` if a checkout provides it,
otherwise download and cache it from the model repo.

One implementation shared by the model loader and the lookup table, rather
than two copies that can drift.
"""

from __future__ import annotations

import os
from importlib.resources import files

#: Hugging Face repo holding per-language directories.
HF_REPO = "soodoku/indicate"
HF_REVISION = "v0.7.0"


def local_data_path(subdir: str, rel: str) -> str:
    """Return the in-package path for a data file, which may not exist.

    Args:
        subdir: Per-language directory under ``indicate/data``.
        rel: Path relative to that directory.

    Returns:
        An absolute filesystem path.
    """
    return os.path.join(str(files("indicate")), "data", subdir, rel)


def resolve_data(
    subdir: str, rel: str, *, repo: str = HF_REPO, revision: str = HF_REVISION
) -> str:
    """Resolve a data file, downloading from Hugging Face if it is not local.

    Args:
        subdir: Per-language directory under ``indicate/data``.
        rel: Path relative to that directory.
        repo: Hugging Face repo id.
        revision: Repo revision to pin.

    Returns:
        A path to the file on disk.
    """
    local = local_data_path(subdir, rel)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, f"{subdir}/{rel}", revision=revision)


def try_resolve_data(
    subdir: str, rel: str, *, repo: str = HF_REPO, revision: str = HF_REVISION
) -> str | None:
    """Resolve a data file, or return ``None`` when it cannot be found.

    Used for optional assets, where absence should disable a feature rather than
    fail a transliteration.

    Args:
        subdir: Per-language directory under ``indicate/data``.
        rel: Path relative to that directory.
        repo: Hugging Face repo id.
        revision: Repo revision to pin.

    Returns:
        A path to the file, or ``None`` if it is neither local nor downloadable.
    """
    try:
        return resolve_data(subdir, rel, repo=repo, revision=revision)
    except Exception:
        return None
