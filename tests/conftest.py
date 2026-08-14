"""Shared fixtures, and an artifact policy that makes a fresh clone honest.

The suite used to be calibrated to one laptop. Model weights (~120 MB) and the
lookup tables are both gitignored, so on any other machine 17 tests skipped
silently and 6 failed for reasons that had nothing to do with the code. Skips
vanish into dots, so nobody noticed.

Three mechanisms fix that:

``needs_lookup`` / ``needs_weights``
    Mark a test with what it actually requires. Detection is offline -- a local
    file, or something already in the Hugging Face cache. It never reaches the
    network, so collection stays fast and works on a plane.

``--require-artifacts``
    Turn those skips into failures. CI runs one leg with it, after building the
    tables from the committed corpora, so "it skipped on the runner" stops being
    a way for coverage to quietly disappear.

A terminal summary
    Names every missing artifact and the exact command that produces it, plus
    how many tests it suppressed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from indicate.languages import PAIRS  # noqa: E402
from indicate.lookup import LOOKUP_FILE  # noqa: E402
from indicate.resources import HF_REPO, HF_REVISION, local_data_path  # noqa: E402
from indicate.transliterator import ENCODER_FILE  # noqa: E402

#: How to produce each missing artifact, quoted back to the developer.
FIX = {
    "lookup": "uv run --group train python training/build_lookup.py --lang {lang}",
    "weights": (
        "weights download from {repo}@{revision} on first use; "
        "check the network or prime the HF cache"
    ),
}


def _cached_on_hf(subdir: str, rel: str) -> bool:
    """Report whether a file is already in the local Hugging Face cache.

    Args:
        subdir: Per-language directory.
        rel: Path within it.

    Returns:
        True if present, without making any network request.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover - huggingface_hub is a hard dep
        return False
    found = try_to_load_from_cache(HF_REPO, f"{subdir}/{rel}", revision=HF_REVISION)
    return isinstance(found, str)


def _available(subdir: str, rel: str) -> bool:
    """Report whether an asset can be had without the network."""
    return os.path.exists(local_data_path(subdir, rel)) or _cached_on_hf(subdir, rel)


def lookup_available(subdir: str) -> bool:
    """Report whether a lookup table can be loaded for a language directory.

    Args:
        subdir: Per-language directory, e.g. ``"punjabi_to_english"``.

    Returns:
        True if the table is on disk or already cached.
    """
    return _available(subdir, LOOKUP_FILE)


def weights_available(subdir: str) -> bool:
    """Report whether model weights can be loaded for a language directory.

    Args:
        subdir: Per-language directory.

    Returns:
        True if the weights are on disk or already cached.
    """
    return _available(subdir, ENCODER_FILE)


def _missing() -> dict[str, list[str]]:
    """Return the missing artifacts, keyed by kind, as language names."""
    missing: dict[str, list[str]] = {"lookup": [], "weights": []}
    for pair in PAIRS.values():
        if not lookup_available(pair.subdir):
            missing["lookup"].append(pair.source)
        if not weights_available(pair.subdir):
            missing["weights"].append(pair.source)
    return missing


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the artifact-strictness flag.

    Args:
        parser: The pytest option parser.
    """
    parser.addoption(
        "--require-artifacts",
        action="store_true",
        default=False,
        help="fail instead of skipping when a lookup table or weights are absent",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Record which artifacts are present, once per session.

    Args:
        config: The pytest config object.
    """
    config.stash_missing = _missing()  # type: ignore[attr-defined]


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip (or fail) tests whose artifacts are absent.

    Args:
        config: The pytest config object.
        items: Collected test items, modified in place.
    """
    missing = config.stash_missing  # type: ignore[attr-defined]
    strict = config.getoption("--require-artifacts")
    for kind, marker in (("lookup", "needs_lookup"), ("weights", "needs_weights")):
        absent = missing[kind]
        if not absent:
            continue
        if strict:
            # Add nothing: the tests run, and the ones that genuinely need the
            # missing artifact fail. That is what "a skip becomes a failure"
            # means. Marking them xfail-strict instead would also flag every
            # test that was marked conservatively but does not really need it,
            # which is noise rather than signal.
            continue
        fix = FIX[kind].format(lang=absent[0], repo=HF_REPO, revision=HF_REVISION)
        reason = f"no {kind} for {', '.join(absent)}; {fix}"
        for item in items:
            if marker in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Say loudly what was missing and how to produce it.

    Args:
        terminalreporter: The pytest terminal reporter.
        exitstatus: The session exit status.
        config: The pytest config object.
    """
    missing = getattr(config, "stash_missing", None)
    if not missing or not any(missing.values()):
        return
    write = terminalreporter.write_line
    terminalreporter.section("artifacts not present", sep="-")
    for kind, langs in missing.items():
        if not langs:
            continue
        fix = FIX[kind].format(lang=langs[0], repo=HF_REPO, revision=HF_REVISION)
        write(f"  {kind}: missing for {', '.join(langs)}")
        write(f"    fix: {fix}")
    write("  Tests needing them were skipped. Use --require-artifacts to fail instead.")


@pytest.fixture
def runner() -> CliRunner:
    """A Click test runner."""
    return CliRunner()


@pytest.fixture
def text_file(tmp_path: Path):
    """Return a factory writing UTF-8 text (or bytes) to a temp file.

    Replaces the ``NamedTemporaryFile(delete=False)`` + ``finally: unlink``
    boilerplate repeated a dozen times across the CLI tests.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A callable ``(content, name=...) -> Path``.
    """
    count = 0

    def write(content: str | bytes, name: str | None = None) -> Path:
        nonlocal count
        count += 1
        path = tmp_path / (name or f"input{count}.txt")
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    return write


@pytest.fixture
def fresh_caches():
    """Drop loaded tables and models. Not autouse -- weight loading is expensive."""
    from indicate import lookup, transliterator

    lookup.clear_cache()
    transliterator.clear_models()
    yield
    lookup.clear_cache()
    transliterator.clear_models()


@pytest.fixture
def isolated_hf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point Hugging Face at an empty cache and forbid the network.

    Without redirecting ``HF_HOME`` an offline test passes on a developer
    machine for the wrong reason: the warm cache answers it.

    Args:
        tmp_path: pytest's per-test temporary directory.
        monkeypatch: pytest's environment patcher.

    Returns:
        The temporary cache directory.
    """
    cache = tmp_path / "hf"
    cache.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return cache
