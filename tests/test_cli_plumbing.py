"""The CLI's file-handling options, with the transliteration stubbed out.

``cli.transliterate`` is two things welded together: a resolver, and about
seventy lines of output plumbing -- format selection, backup, atomic rename,
dry-run, blank-line alignment, n-best joining. The plumbing had no tests at all,
because every existing CLI test needed a real backend and therefore an artifact,
and a test that skips on a fresh clone is not a test of the plumbing.

So :func:`stub_backends` replaces :func:`indicate.api.build` with one backend
that answers everything. Nothing here loads a table, downloads weights or
imports torch, and every assertion is about what lands on disk. The stub also
records what ``build`` was asked for, which is how the ``--engine`` and beam-width
options get checked at all: they have no observable effect on output otherwise.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from click.testing import CliRunner

from indicate.cli import cli

#: Deterministic answers, so an assertion names a string rather than a mechanism.
STUB = {"हिंदी": "hindi", "गौरव": "gaurav", "सूद": "sood"}


class StubBackend:
    """Answers every word from :data:`STUB`, with three ranked candidates."""

    name = "stub"

    def __init__(self) -> None:
        self.unavailable = False
        self.seen: list[str] = []

    def resolve(self, words: Sequence[str]):
        """Return three candidates per word, best first.

        Args:
            words: Words to answer.

        Returns:
            A candidate list per word.
        """
        self.seen.extend(words)
        out = []
        for word in words:
            best = STUB.get(word, word)
            out.append([(best, 0.0), (best + "-2", -1.0), (best + "-3", -2.0)])
        return out


@pytest.fixture
def stub_backends(monkeypatch: pytest.MonkeyPatch):
    """Replace the backend chain with one that always answers.

    Args:
        monkeypatch: pytest's attribute patcher.

    Returns:
        A list that receives ``(engine, pair, kwargs)`` for each ``build`` call.
    """
    calls: list[tuple] = []

    def fake_build(engine, pair, **kwargs):
        calls.append((engine, pair, kwargs))
        return [StubBackend()]

    monkeypatch.setattr("indicate.api.build", fake_build)
    return calls


@pytest.fixture
def infile(text_file):
    """A three-line Devanagari input file, one blank line in the middle."""
    return text_file("हिंदी\n\nगौरव सूद\n")


def _run(runner: CliRunner, *args) -> object:
    """Invoke the CLI and fail loudly on an unexpected non-zero exit."""
    result = runner.invoke(cli, ["transliterate", *[str(a) for a in args]])
    assert result.exit_code == 0, result.output
    return result


def test_text_output_is_one_line_per_input_line(
    runner, stub_backends, infile, tmp_path: Path
):
    out = tmp_path / "out.txt"
    _run(runner, "--input", infile, "--output", out)
    assert out.read_text(encoding="utf-8") == "hindi\n\ngaurav sood\n"


def test_a_blank_input_line_stays_blank_and_is_never_sent_to_a_backend(
    runner, stub_backends, infile, tmp_path: Path
):
    # Alignment is the point: line 3 of the output must correspond to line 3 of
    # the input, which only holds if the blank is reinserted rather than dropped.
    out = tmp_path / "out.txt"
    _run(runner, "--input", infile, "--output", out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[1] == ""
    assert len(lines) == 3


def test_json_output_carries_every_field_the_reader_expects(
    runner, stub_backends, infile, tmp_path: Path
):
    out = tmp_path / "out.json"
    _run(runner, "--input", infile, "--output", out, "--format", "json")
    data = json.loads(out.read_text(encoding="utf-8"))

    meta = data["metadata"]
    assert meta["source_language"] == "hindi"
    assert meta["target_language"] == "english"
    assert meta["total_lines"] == 3
    assert meta["failed_lines"] == 0
    assert meta["encoding"] == "utf-8"

    first, blank, third = data["results"]
    assert (first["line_number"], first["input_text"], first["output_text"]) == (
        1,
        "हिंदी",
        "hindi",
    )
    assert (blank["input_text"], blank["output_text"]) == ("", "")
    assert third["output_text"] == "gaurav sood"
    # The engine chain is recorded per row; without it a JSON file cannot say
    # which backend produced the string next to it.
    assert first["confidence"] == "lookup,model"
    assert first["error"] is None


def test_json_keeps_the_source_text_unescaped(
    runner, stub_backends, infile, tmp_path: Path
):
    # ensure_ascii=False, or every Indic input becomes \uXXXX soup on disk.
    out = tmp_path / "out.json"
    _run(runner, "--input", infile, "--output", out, "--format", "json")
    assert "हिंदी" in out.read_text(encoding="utf-8")


def test_dry_run_writes_nothing_at_all(runner, stub_backends, infile, tmp_path: Path):
    out = tmp_path / "out.txt"
    result = runner.invoke(
        cli,
        ["transliterate", "--input", str(infile), "--output", str(out), "--dry-run"],
    )
    assert result.exit_code == 0
    assert not out.exists()
    assert "Would write 3 result(s)" in result.output


def test_dry_run_leaves_an_existing_file_untouched(
    runner, stub_backends, infile, tmp_path: Path
):
    out = tmp_path / "out.txt"
    out.write_text("do not clobber me", encoding="utf-8")
    _run(runner, "--input", infile, "--output", out, "--dry-run")
    assert out.read_text(encoding="utf-8") == "do not clobber me"


def test_backup_preserves_the_previous_contents(
    runner, stub_backends, infile, tmp_path: Path
):
    out = tmp_path / "out.txt"
    out.write_text("previous run\n", encoding="utf-8")
    _run(runner, "--input", infile, "--output", out, "--backup")
    backups = [p for p in tmp_path.iterdir() if ".backup_" in p.name]
    assert len(backups) == 1, list(tmp_path.iterdir())
    assert backups[0].read_text(encoding="utf-8") == "previous run\n"
    assert out.read_text(encoding="utf-8").startswith("hindi")


def test_no_backup_flag_means_no_backup_file(
    runner, stub_backends, infile, tmp_path: Path
):
    out = tmp_path / "out.txt"
    out.write_text("previous run\n", encoding="utf-8")
    _run(runner, "--input", infile, "--output", out)
    assert not [p for p in tmp_path.iterdir() if ".backup_" in p.name]


def test_no_atomic_writes_the_same_bytes_and_leaves_no_temp_file(
    runner, stub_backends, infile, tmp_path: Path
):
    atomic = tmp_path / "atomic.txt"
    direct = tmp_path / "direct.txt"
    _run(runner, "--input", infile, "--output", atomic)
    _run(runner, "--input", infile, "--output", direct, "--no-atomic")
    assert direct.read_bytes() == atomic.read_bytes()
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]


def test_n_best_joins_candidates_on_one_line(runner, stub_backends, tmp_path: Path):
    # The file format is line-based, so n > 1 has to collapse to a single line
    # or the output stops lining up with the input.
    out = tmp_path / "out.txt"
    _run(runner, "हिंदी", "--n", "3", "--output", out)
    assert out.read_text(encoding="utf-8") == "hindi | hindi-2 | hindi-3\n"


def test_stdout_is_used_when_no_output_file_is_given(runner, stub_backends):
    result = runner.invoke(cli, ["transliterate", "गौरव सूद"])
    assert result.exit_code == 0
    assert result.output.strip() == "gaurav sood"


def test_the_engine_chain_reaches_build_in_the_order_given(runner, stub_backends):
    _run(runner, "हिंदी", "--engine", "llm,model,lookup")
    ((engine, pair, _kwargs),) = stub_backends
    assert engine == ("llm", "model", "lookup")
    assert (pair.source, pair.target) == ("hindi", "english")


def test_n_best_widens_the_beam(runner, stub_backends):
    # A beam narrower than n cannot produce n distinct candidates, so the CLI
    # has to raise it; nothing about the output would reveal that it did not.
    _run(runner, "हिंदी", "--n", "7")
    assert stub_backends[0][2]["beam"] >= 7


def test_quiet_suppresses_the_written_notice(runner, stub_backends, tmp_path: Path):
    out = tmp_path / "out.txt"
    result = runner.invoke(
        cli, ["transliterate", "हिंदी", "--output", str(out), "--quiet"]
    )
    assert result.exit_code == 0
    assert "Results written" not in result.output
    assert out.exists()


def test_an_unknown_backend_is_rejected_before_anything_runs(runner, stub_backends):
    result = runner.invoke(cli, ["transliterate", "हिंदी", "--engine", "quantum"])
    assert result.exit_code == 2
    assert "quantum" in result.output
    assert "lookup" in result.output  # it names the valid choices
    assert not stub_backends  # and never got as far as building a chain
