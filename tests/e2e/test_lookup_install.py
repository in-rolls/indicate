"""The documented lookup-table workflow, run against a real installed wheel.

The README tells a user to build a table with ``training/build_lookup.py``.
Nothing verified that an *installed* package could then find it, and it could
not: :func:`indicate.resources.local_data_path` resolved only to
``site-packages/indicate/data/``, while the build script writes into whatever
checkout it was run from. The documented instruction ended at a file the
package would never look at, and the only way to complete it was to copy a file
into site-packages by hand -- which an upgrade silently discards.

``INDICATE_DATA_DIR`` closes that. These tests run the whole path in one go:
install the wheel into a clean venv, confirm the lookup backend is genuinely
absent, point the variable at a directory holding a real table, and confirm the
same call now answers from it -- without importing torch, which is the entire
point of the backend.

Every assertion runs in a subprocess against the venv interpreter, never this
one, so the source tree cannot stand in for the installed package.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from indicate.languages import PAIRS
from indicate.lookup import LOOKUP_FILE
from indicate.resources import local_data_path

from ..helpers import run_python
from .conftest import venv_python

PA = PAIRS[("punjabi", "english")]
SINGH = "ਸਿੰਘ"

#: One word per local pair, and what its table answers. Both languages, because
#: the workflow is documented for both and testing one proves nothing about the
#: other -- they are separate tables, separate subdirectories, separate builds.
WORDS = {
    ("punjabi", "english"): (SINGH, "singh"),
    ("hindi", "english"): ("सिंह", "singh"),
}


@pytest.fixture(params=sorted(WORDS), ids=lambda key: key[0])
def language(request) -> tuple:
    """Each local pair in turn, as ``(pair, word, expected)``."""
    pair = PAIRS[request.param]
    word, expected = WORDS[request.param]
    return pair, word, expected


@pytest.fixture
def real_table(tmp_path: Path, language: tuple) -> Path:
    """A data directory holding a genuine built table, laid out as expected.

    Uses the developer's table rather than a synthetic one so the layout under
    test is the layout ``build_lookup.py`` actually produces.

    Args:
        tmp_path: pytest's per-test temporary directory.
        language: The pair under test.

    Returns:
        A directory to point ``INDICATE_DATA_DIR`` at.
    """
    pair = language[0]
    built = local_data_path(pair.subdir, LOOKUP_FILE)
    if not Path(built).exists():
        pytest.skip(
            f"no built table at {built}; "
            f"run training/build_lookup.py --lang {pair.source}"
        )
    root = tmp_path / "indicate-data"
    (root / pair.subdir).mkdir(parents=True)
    shutil.copy2(built, root / pair.subdir / LOOKUP_FILE)
    return root


def _script(body: str) -> str:
    return "import sys\n" + body


def test_the_installed_wheel_has_no_table_and_says_so(wheel_venv: Path, clean_env):
    proc = run_python(
        _script(
            "from indicate.lookup import Lookup\n"
            f"print(Lookup.load({PA.subdir!r}) is None)\n"
        ),
        executable=str(venv_python(wheel_venv)),
        cwd=wheel_venv,
        env=clean_env,
    )
    assert proc.stdout.strip() == "True"


def test_a_user_built_table_is_found_through_the_env_var(
    wheel_venv: Path, clean_env, real_table: Path, language: tuple
):
    # The workflow the README documents, end to end: pip install, build a
    # table elsewhere, point the variable at it, get table answers. Runs for
    # every local pair -- they are separate tables in separate directories, so
    # Punjabi working says nothing about Hindi.
    pair, word, expected = language
    env = {**clean_env, "INDICATE_DATA_DIR": str(real_table)}
    proc = run_python(
        _script(
            "import indicate\n"
            f"out = indicate.transliterate({word!r}, source={pair.source!r}, "
            "engine=['lookup'])\n"
            "print(out, 'torch' in sys.modules)\n"
        ),
        executable=str(venv_python(wheel_venv)),
        cwd=wheel_venv,
        env=env,
    )
    answer, torch_imported = proc.stdout.split()
    assert answer == expected
    # The reason to have a table at all: a hit never reaches the decoder.
    assert torch_imported == "False"


def test_languages_reports_the_backend_ready_once_it_is_pointed_at_one(
    wheel_venv: Path, clean_env, real_table: Path, language: tuple
):
    # `indicate languages` is what a user runs to find out whether the setup
    # worked, so it has to change its answer.
    args = (
        "sys.argv = ['indicate', 'languages']\nfrom indicate.cli import main\nmain()\n"
    )
    without = run_python(
        _script(args),
        executable=str(venv_python(wheel_venv)),
        cwd=wheel_venv,
        env=clean_env,
        check=False,
    )
    with_table = run_python(
        _script(args),
        executable=str(venv_python(wheel_venv)),
        cwd=wheel_venv,
        env={**clean_env, "INDICATE_DATA_DIR": str(real_table)},
        check=False,
    )

    source = language[0].source

    def lookup_row(out: str) -> str:
        # Only the pair under test: the fixture supplies exactly one table, so
        # the other direction must stay unavailable. Asserting over every row
        # would demand a table this test never provided.
        rows = [
            ln for ln in out.splitlines() if ln.startswith(source) and "lookup" in ln
        ]
        assert len(rows) == 1, out
        return rows[0]

    assert "unavailable" in lookup_row(without.stdout)
    assert "ready" in lookup_row(with_table.stdout)
    # And the untouched direction is unchanged, so this is the variable taking
    # effect rather than the status logic collapsing to "everything is fine".
    assert "unavailable" in with_table.stdout


def test_the_env_var_does_not_have_to_exist_or_be_complete(
    wheel_venv: Path, clean_env, tmp_path: Path
):
    # Pointing it at an empty or missing directory must degrade to the packaged
    # location, not raise. Otherwise a stale variable bricks the install.
    for target in (tmp_path / "empty", tmp_path / "does-not-exist"):
        target.mkdir(exist_ok=True)
        proc = run_python(
            _script(
                "from indicate.lookup import Lookup\n"
                f"print(Lookup.load({PA.subdir!r}) is None)\n"
            ),
            executable=str(venv_python(wheel_venv)),
            cwd=wheel_venv,
            env={**clean_env, "INDICATE_DATA_DIR": str(target)},
        )
        assert proc.stdout.strip() == "True"
