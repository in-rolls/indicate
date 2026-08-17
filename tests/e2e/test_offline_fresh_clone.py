"""What a fresh install with no network actually does.

This is the configuration nobody develops in and everybody installs into: no
weights, no lookup table, no route to huggingface.co. Until
:class:`~indicate.BackendsUnavailableError` existed, the answer was ``""`` and
exit code 0 -- an empty string is a plausible transliteration of nothing, so it
propagated into whatever the caller did next.

Two details make this test mean what it says:

* it runs in a **subprocess** with ``HF_HOME`` pointed at an empty directory.
  Inheriting a developer's warm cache would make it pass for the wrong reason,
  and the cache is process-global, so an in-process test cannot undo it;
* it **skips when the source tree actually has the artifacts**. Those are real
  files inside ``indicate/data/``; no environment variable hides them, and
  pretending otherwise would just assert that the mocking worked.

The companion assertion -- that the *installed wheel* has no such files to find
-- lives in ``test_wheel_install.py``, which is the same scenario with the skip
condition guaranteed false.
"""

from __future__ import annotations

import os
import sys

import pytest

from indicate.languages import PAIRS, supports
from indicate.lookup import LOOKUP_FILE
from indicate.resources import local_data_path
from indicate.transliterator import ENCODER_FILE

from ..helpers import run_python

HINDI = "राजशेखर"


def _artifacts_in_source_tree() -> list[str]:
    """Return the artifact paths a checkout provides, which env vars cannot hide."""
    present = []
    for pair in PAIRS.values():
        rels = [LOOKUP_FILE]
        if supports(*pair.key, "model"):
            rels.append(ENCODER_FILE)
        for rel in rels:
            path = local_data_path(pair.subdir, rel)
            if os.path.exists(path):
                present.append(path)
    return present


def _offline_env(tmp_path) -> dict[str, str]:
    """Return an environment with an empty HF cache and the network forbidden."""
    env = dict(os.environ)
    cache = tmp_path / "hf"
    cache.mkdir(exist_ok=True)
    env["HF_HOME"] = str(cache)
    env["HF_HUB_OFFLINE"] = "1"
    # HF_HOME is the modern knob; the older ones still win where they are set,
    # and a developer machine may well have them.
    env.pop("HUGGINGFACE_HUB_CACHE", None)
    env.pop("TRANSFORMERS_CACHE", None)
    return env


@pytest.fixture
def offline(tmp_path):
    """An offline child environment, or a skip when the checkout has artifacts."""
    present = _artifacts_in_source_tree()
    if present:
        pytest.skip(
            f"source tree provides {len(present)} artifact(s), e.g. {present[0]}"
        )
    return _offline_env(tmp_path)


def test_a_fresh_install_offline_raises_rather_than_returning_empty(offline):
    proc = run_python(
        "import indicate\n"
        "try:\n"
        f"    out = indicate.transliterate({HINDI!r}, source='hindi')\n"
        "except indicate.BackendsUnavailableError as exc:\n"
        "    print('RAISED', exc)\n"
        "else:\n"
        "    print('RETURNED', repr(out))\n",
        env=offline,
    )
    assert proc.stdout.startswith("RAISED"), proc.stdout
    # The message has to say what to do about it, not just that it happened.
    assert "build_lookup" in proc.stdout
    assert "weights" in proc.stdout


def test_the_cli_exits_one_instead_of_printing_a_blank_line(offline):
    proc = run_python(
        "import sys\n"
        "from indicate.cli import main\n"
        f"sys.argv = ['indicate', 'transliterate', {HINDI!r}]\n"
        "main()\n",
        env=offline,
        check=False,
    )
    assert proc.returncode == 1, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert proc.stdout.strip() == ""
    # `in`, not `startswith`: the backends log why they declined first.
    assert "Error: nothing could answer" in proc.stderr


def test_languages_still_runs_and_reports_the_backends_as_unavailable(offline):
    # The command that answers "why did that fail" must not itself fail.
    proc = run_python(
        "import sys\n"
        "from indicate.cli import main\n"
        "sys.argv = ['indicate', 'languages']\n"
        "main()\n",
        env=offline,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "unavailable" in proc.stdout
    assert "build_lookup.py" in proc.stdout


def test_nothing_heavy_is_imported_just_to_fail(offline):
    # Reporting "no backend could run" must not cost a 0.4s torch import.
    proc = run_python(
        "import sys\n"
        "import indicate\n"
        "try:\n"
        f"    indicate.transliterate({HINDI!r}, source='hindi', engine=['lookup'])\n"
        "except indicate.BackendsUnavailableError:\n"
        "    pass\n"
        "print('torch' in sys.modules, 'litellm' in sys.modules)\n",
        env=offline,
    )
    assert proc.stdout.split() == ["False", "False"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__]))
