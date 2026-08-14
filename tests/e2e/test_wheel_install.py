"""The installed package, tested as installed.

CI's existing wheel job installs the wheel and then runs ``pytest tests/`` from
the repo root, where ``tests/__init__.py`` plus pytest's prepend import mode make
``import indicate`` resolve to the **source tree**. It has therefore never tested
a wheel — and the source tree carries lookup tables and weights the wheel does
not, which is exactly the class of defect that arrangement will miss.

Three independent guards here, because the failure mode is that it *looks* like
it worked: the suite is never run from the repo root, every subprocess runs with
``cwd`` outside the checkout and ``PYTHONPATH`` scrubbed, and the first test
asserts where ``indicate.__file__`` actually resolved.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.e2e.conftest import REPO_ROOT, venv_python


def _run(venv: Path, body: str, env: dict[str, str], cwd: Path):
    return subprocess.run(  # noqa: S603
        [str(venv_python(venv)), "-c", body],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        check=False,
    )


class TestItIsTheWheelUnderTest:
    def test_indicate_resolves_inside_the_venv_not_the_repo(
        self, wheel_venv: Path, clean_env: dict[str, str], tmp_path: Path
    ):
        # If a future change reintroduces source-tree shadowing, this fails
        # loudly instead of silently passing every other test in this file.
        proc = _run(
            wheel_venv, "import indicate; print(indicate.__file__)", clean_env, tmp_path
        )
        assert proc.returncode == 0, proc.stderr
        resolved = Path(proc.stdout.strip())
        assert "site-packages" in resolved.parts, resolved
        assert REPO_ROOT not in resolved.parents, (
            f"import resolved to the source tree at {resolved}, not the wheel"
        )


class TestImportHygiene:
    def test_importing_the_package_pulls_in_no_heavy_dependency(
        self, wheel_venv: Path, clean_env: dict[str, str], tmp_path: Path
    ):
        # Only click is installed, so this is a real assertion rather than a
        # hopeful one: if any module grew a top-level torch import, the import
        # would fail outright here.
        body = (
            "import sys, indicate\n"
            "print(','.join(m for m in ('torch','numpy','litellm','safetensors') "
            "if m in sys.modules) or 'none')\n"
        )
        proc = _run(wheel_venv, body, clean_env, tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "none"

    def test_the_cli_module_also_imports_clean(
        self, wheel_venv: Path, clean_env: dict[str, str], tmp_path: Path
    ):
        body = (
            "import sys, indicate.cli\n"
            "assert 'torch' not in sys.modules\n"
            "assert 'litellm' not in sys.modules\n"
            "print('ok')\n"
        )
        proc = _run(wheel_venv, body, clean_env, tmp_path)
        assert proc.returncode == 0, proc.stderr


class TestInstalledDataFiles:
    def test_tokenizers_are_present_and_tables_are_not(
        self, wheel_venv: Path, clean_env: dict[str, str], tmp_path: Path
    ):
        body = (
            "from importlib.resources import files\n"
            "import pathlib\n"
            "root = pathlib.Path(str(files('indicate'))) / 'data'\n"
            "print((root / 'hindi_to_english' / 'hindi_tokens.json').is_file(),\n"
            "      (root / 'hindi_to_english' / 'lookup.tsv.gz').is_file(),\n"
            "      (root / 'hindi_to_english' / 'saved_weights').is_dir())\n"
        )
        proc = _run(wheel_venv, body, clean_env, tmp_path)
        assert proc.returncode == 0, proc.stderr
        tokenizer, table, weights = proc.stdout.split()
        assert tokenizer == "True"
        assert table == "False"
        assert weights == "False"


class TestFreshInstallBehaviour:
    def test_with_nothing_available_it_raises_instead_of_returning_empty(
        self, wheel_venv: Path, clean_env: dict[str, str], tmp_path: Path
    ):
        # This is the honest fresh-install experience, pinned. Before the
        # availability fix it returned "" and exited 0.
        body = (
            "import indicate\n"
            "from indicate import BackendsUnavailableError\n"
            "try:\n"
            "    out = indicate.transliterate('राजशेखर', source='hindi')\n"
            "except BackendsUnavailableError as exc:\n"
            "    print('RAISED', 'lookup' in str(exc), 'model' in str(exc))\n"
            "else:\n"
            "    print('RETURNED', repr(out))\n"
        )
        proc = _run(wheel_venv, body, clean_env, tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.split() == ["RAISED", "True", "True"], proc.stdout

    def test_an_unsupported_direction_names_the_alternative(
        self, wheel_venv: Path, clean_env: dict[str, str], tmp_path: Path
    ):
        body = (
            "import indicate\n"
            "try:\n"
            "    indicate.transliterate('வணக்கம்', source='tamil')\n"
            "except indicate.UnsupportedPairError as exc:\n"
            "    print('llm' in str(exc))\n"
        )
        proc = _run(wheel_venv, body, clean_env, tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "True"
