"""The ``indicate`` console script, run as a real process.

Nothing had ever executed it. A typo in ``[project.scripts]`` would have shipped:
``twine check`` validates metadata, and CI's ``python -c "import indicate"``
never touches ``cli``.

A real process also reaches things ``CliRunner`` structurally cannot — stdout
and stderr are genuinely separate streams, and ``sys.stdin.isatty()`` is
genuinely false when piped, which is the branch the CLI's own progress-banner
guard exists for.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.e2e.conftest import venv_script


@pytest.fixture(scope="session")
def script(wheel_venv: Path) -> Path:
    """The installed console script, resolved from the venv, not from PATH."""
    path = venv_script(wheel_venv, "indicate")
    if not path.exists():
        pytest.skip(f"console script was not installed at {path}")
    return path


def run(script: Path, *args: str, env=None, cwd: Path, stdin: str | None = None):
    """Invoke the console script and return the completed process."""
    return subprocess.run(  # noqa: S603
        [str(script), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        input=stdin,
        check=False,
    )


class TestTheScriptExists:
    def test_version_prints_something_versionlike(
        self, script: Path, clean_env, tmp_path: Path
    ):
        proc = run(script, "--version", env=clean_env, cwd=tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert any(ch.isdigit() for ch in proc.stdout)

    def test_help_lists_every_command(self, script: Path, clean_env, tmp_path: Path):
        proc = run(script, "--help", env=clean_env, cwd=tmp_path)
        assert proc.returncode == 0, proc.stderr
        for command in ("transliterate", "languages", "info"):
            assert command in proc.stdout

    def test_an_unknown_command_exits_two(
        self, script: Path, clean_env, tmp_path: Path
    ):
        proc = run(script, "nosuchcommand", env=clean_env, cwd=tmp_path)
        assert proc.returncode == 2
        # The old per-language commands are gone; make sure that is what happens.
        proc = run(script, "hindi2english", "x", env=clean_env, cwd=tmp_path)
        assert proc.returncode == 2


class TestLanguagesCommand:
    def test_it_reports_the_local_directions(
        self, script: Path, clean_env, tmp_path: Path
    ):
        # First test of this command's body at all.
        proc = run(script, "languages", env=clean_env, cwd=tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert "hindi -> english" in proc.stdout
        assert "punjabi -> english" in proc.stdout

    def test_it_works_offline(self, script: Path, clean_env, tmp_path: Path):
        # clean_env sets HF_HUB_OFFLINE=1 and an empty cache.
        proc = run(script, "languages", env=clean_env, cwd=tmp_path)
        assert proc.returncode == 0, proc.stderr

    def test_it_distinguishes_downloadable_and_locally_built_lookups(
        self, script: Path, clean_env, tmp_path: Path
    ):
        # This venv has no lookup tables. Bengali is published and can be
        # fetched later; the restricted Hindi and Punjabi tables cannot.
        proc = run(script, "languages", env=clean_env, cwd=tmp_path)
        assert proc.returncode == 0, proc.stderr
        # Table rows only: the trailing hint also mentions the word "lookup".
        rows = [
            line
            for line in proc.stdout.splitlines()
            if line.split()[-1:] and "lookup" in line.split()
            if line.rstrip().endswith(("ready", "unavailable", "first use"))
        ]
        assert rows, proc.stdout
        by_language = {
            row.split(" ->", 1)[0].strip(): row for row in rows if " ->" in row
        }
        assert by_language["bengali"].endswith("downloads on first use")
        assert by_language["hindi"].endswith("unavailable")
        assert by_language["punjabi"].endswith("unavailable")
        assert "build_lookup.py" in proc.stdout


class TestInfoCommand:
    def test_it_names_the_weights_repo_and_works_offline(
        self, script: Path, clean_env, tmp_path: Path
    ):
        proc = run(script, "info", env=clean_env, cwd=tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert "gojiberries/indicate" in proc.stdout
        assert "hindi -> english" in proc.stdout


class TestTransliterateCommand:
    def test_no_input_at_all_fails_with_a_message_on_stderr(
        self, script: Path, clean_env, tmp_path: Path
    ):
        # A real process separates the streams; CliRunner merges them, so this
        # assertion is only possible here.
        proc = run(script, "transliterate", env=clean_env, cwd=tmp_path, stdin="")
        assert proc.returncode == 1
        assert "No text to transliterate" in proc.stderr
        assert proc.stdout.strip() == ""

    def test_an_unknown_backend_is_refused_by_name(
        self, script: Path, clean_env, tmp_path: Path
    ):
        proc = run(
            script,
            "transliterate",
            "x",
            "--engine",
            "quantum",
            env=clean_env,
            cwd=tmp_path,
        )
        assert proc.returncode == 2
        assert "quantum" in proc.stderr
        for known in ("lookup", "model", "llm"):
            assert known in proc.stderr

    def test_an_unsupported_direction_is_refused_not_billed(
        self, script: Path, clean_env, tmp_path: Path
    ):
        proc = run(script, "transliterate", "வணக்கம்", env=clean_env, cwd=tmp_path)
        assert proc.returncode == 1
        assert "tamil" in proc.stderr
        assert "llm" in proc.stderr

    def test_with_no_artifacts_it_fails_loudly_rather_than_printing_nothing(
        self, script: Path, clean_env, tmp_path: Path
    ):
        proc = run(script, "transliterate", "राजशेखर", env=clean_env, cwd=tmp_path)
        assert proc.returncode == 1, (
            f"expected a loud failure, got exit {proc.returncode} "
            f"with stdout {proc.stdout!r}"
        )
        assert "Error" in proc.stderr
        assert proc.stdout.strip() == ""

    def test_it_reads_a_file(self, script: Path, clean_env, tmp_path: Path):
        source = tmp_path / "in.txt"
        source.write_text("राजशेखर\nगौरव\n", encoding="utf-8")
        proc = run(
            script,
            "transliterate",
            "--input",
            str(source),
            env=clean_env,
            cwd=tmp_path,
        )
        # No artifacts in this venv, so the honest outcome is a loud failure --
        # what matters is that the file was read and the path did not crash.
        assert proc.returncode == 1
        assert "Error" in proc.stderr

    def test_a_missing_input_file_is_click_s_error(
        self, script: Path, clean_env, tmp_path: Path
    ):
        proc = run(
            script,
            "transliterate",
            "--input",
            "nope.txt",
            env=clean_env,
            cwd=tmp_path,
        )
        assert proc.returncode == 2
        assert "nope.txt" in proc.stderr
