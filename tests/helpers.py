"""Shared test helpers.

Two things live here. The per-language shims that seven test files had each
copy-pasted, and a **named vocabulary** that separates words by which backend
answers them.

That second part is the important one. A test that asserts
``transliterate("राजशेखर") == "rajshekhar"`` is asserting two mechanisms at once
— that the table has an entry *and* that the chain prefers it — while naming
neither. On a machine without the table it fails, and the failure says nothing
about which half broke. Tests that only need *some* correct output should draw
from :data:`AGREED`, where both backends produce the same string and the
assertion holds in every world.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import indicate

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Words the table and the model romanize identically. Use these wherever the
#: test is about plumbing -- file reading, batching, CLI wiring -- so it does
#: not silently become a test of which artifacts happen to be installed.
AGREED = {
    "हिंदी": "hindi",
    "गौरव": "gaurav",
    "सूद": "sood",
}

#: Words where the two disagree. These are the whole regression surface of the
#: lookup backend, so they belong in tests that are explicitly about it.
TABLE_ONLY = {"राजशेखर": "rajshekhar"}
MODEL_ONLY = {"राजशेखर": "rajshekar"}


def hi(text, **kwargs):
    """Transliterate Hindi through the public API.

    Args:
        text: Devanagari text.
        **kwargs: Passed to :func:`indicate.transliterate`.

    Returns:
        The romanization, or a candidate list when ``n > 1``.
    """
    return indicate.transliterate(text, source="hindi", **kwargs)


def hi_batch(texts, **kwargs):
    """Batch-transliterate Hindi through the public API.

    Args:
        texts: Devanagari texts.
        **kwargs: Passed to :func:`indicate.transliterate_batch`.

    Returns:
        One result per input.
    """
    return indicate.transliterate_batch(texts, source="hindi", **kwargs)


def pa(text, **kwargs):
    """Transliterate Punjabi through the public API.

    Args:
        text: Gurmukhi text.
        **kwargs: Passed to :func:`indicate.transliterate`.

    Returns:
        The romanization, or a candidate list when ``n > 1``.
    """
    return indicate.transliterate(text, source="punjabi", **kwargs)


def pa_batch(texts, **kwargs):
    """Batch-transliterate Punjabi through the public API.

    Args:
        texts: Gurmukhi texts.
        **kwargs: Passed to :func:`indicate.transliterate_batch`.

    Returns:
        One result per input.
    """
    return indicate.transliterate_batch(texts, source="punjabi", **kwargs)


def run_python(
    body: str,
    *,
    executable: str | None = None,
    cwd: Path | str = REPO_ROOT,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a snippet in a fresh interpreter and return the completed process.

    A subprocess rather than an in-process call for two reasons: pytest runs
    under coverage, whose tracer costs roughly 6x on a tight loop, and only a
    fresh interpreter can answer questions like "was torch imported".

    Args:
        body: Python source to execute.
        executable: Interpreter to use; defaults to the current one.
        cwd: Working directory. Point this outside the repo when testing an
            installed package, or the source tree shadows it.
        env: Full environment for the child, or ``None`` to inherit.
        check: Raise :class:`AssertionError` carrying the child's stderr when
            it exits non-zero.

    Returns:
        The completed process, stdout and stderr decoded.

    Raises:
        AssertionError: If ``check`` and the child failed.
    """
    proc = subprocess.run(  # noqa: S603
        [executable or sys.executable, "-c", body],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"child exited {proc.returncode}:\n{proc.stderr}")
    return proc
