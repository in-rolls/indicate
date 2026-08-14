#!/usr/bin/env python3
"""Command line interface.

One transliteration command, because the language and the backend are arguments
rather than separate programs::

    indicate transliterate "राजशेखर चिंतालपति"
    indicate transliterate "ਰਵਿ ਸ਼ਰਮਾ" --from punjabi --engine lookup
    indicate transliterate --input names.txt --output roman.txt --format json
    indicate transliterate "नमस्ते" --engine lookup,llm --provider openai

This replaces ``hindi2english``, ``punjabi2english`` and ``llm``. The language
used to come out of the command name -- which meant feeding Gurmukhi to
``hindi2english`` silently produced garbage -- and the backend came out of
``--lookup/--no-lookup``, which could not express a chain of more than two.
"""

from __future__ import annotations

import sys
import time
from importlib import metadata
from pathlib import Path
from typing import IO, Any

import click

from .engine import DEFAULT_ENGINE, KNOWN, BackendsUnavailableError
from .file_utils import (
    OutputFormat,
    TransliterationResult,
    create_backup,
    read_input_file,
    validate_file_paths,
    write_output_safely,
)
from .languages import (
    LANGUAGES,
    PAIRS,
    UNAVAILABLE,
    UnsupportedPairError,
    resolve_pair,
    status,
    supported,
)


def _get_version() -> str:
    """Return the installed package version, or ``"unknown"``."""
    try:
        return metadata.version("indicate")
    except Exception:
        return "unknown"


class EngineChain(click.ParamType):
    """A comma-separated backend chain, validated against the known names."""

    name = "engine"

    def convert(self, value, param, ctx) -> tuple[str, ...]:
        """Parse and validate ``lookup,model`` into a tuple.

        Args:
            value: The raw option text.
            param: The Click parameter, for error reporting.
            ctx: The Click context, for error reporting.

        Returns:
            The chain, in order.
        """
        if isinstance(value, tuple):
            return value
        names = tuple(part.strip() for part in str(value).split(",") if part.strip())
        if not names:
            self.fail("engine must name at least one backend", param, ctx)
        unknown = [n for n in names if n not in KNOWN]
        if unknown:
            self.fail(
                f"unknown backend(s) {', '.join(unknown)}; "
                f"choose from {', '.join(sorted(KNOWN))}",
                param,
                ctx,
            )
        return names


@click.group()
@click.version_option(version=_get_version())
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Transliterate Indic text to and from other scripts."""
    ctx.ensure_object(dict)


@cli.command()
@click.argument("text", required=False)
@click.option(
    "--from",
    "-f",
    "source",
    default=None,
    help="Source language; auto-detected from the script when omitted",
)
@click.option("--to", "-t", "target", default="english", help="Target language")
@click.option(
    "--engine",
    "-e",
    type=EngineChain(),
    default=",".join(DEFAULT_ENGINE),
    show_default=True,
    help="Backends to try in order: lookup, model, llm",
)
@click.option(
    "--input",
    "-i",
    "input_file",
    type=click.File("r", encoding="utf-8"),
    help="Read text from a file instead of the argument",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(path_type=Path),
    help="Write results to a file instead of stdout",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(OutputFormat.VALID_FORMATS),
    default=OutputFormat.TEXT,
    help="Output format when writing to a file",
)
@click.option("--n", "n_best", type=int, default=1, help="Candidates per input")
@click.option("--provider", "-p", default=None, help="LLM provider for the llm backend")
@click.option("--model", "-m", "llm_model", default=None, help="LLM model name")
@click.option("--api-key", default=None, help="LLM API key; else read from the env")
@click.option("--backup", is_flag=True, help="Back up the output file before writing")
@click.option(
    "--atomic/--no-atomic", default=True, help="Write the output file atomically"
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be written, write nothing"
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def transliterate(
    text: str | None,
    source: str | None,
    target: str,
    engine: tuple[str, ...],
    input_file: IO[str] | None,
    output_file: Path | None,
    output_format: str,
    n_best: int,
    provider: str | None,
    llm_model: str | None,
    api_key: str | None,
    backup: bool,
    atomic: bool,
    dry_run: bool,
    quiet: bool,
) -> None:
    r"""Transliterate TEXT, a file, or standard input.

    Examples:
        indicate transliterate "राजशेखर चिंतालपति"

        indicate transliterate "ਰਵਿ ਸ਼ਰਮਾ" --from punjabi --engine lookup

        indicate transliterate --input names.txt --output roman.txt --format json

    \f

    Args:
        text: Text to transliterate; omit to read from ``--input`` or stdin.
        source: Source language; auto-detected when omitted.
        target: Target language.
        engine: Backends to try in order.
        input_file: Read input from here instead of the argument.
        output_file: Write results here instead of stdout.
        output_format: ``text`` or ``json``, for file output.
        n_best: Number of candidates per input.
        provider: LLM provider for the ``llm`` backend.
        llm_model: LLM model name.
        api_key: LLM API key.
        backup: Back up an existing output file first.
        atomic: Write the output file atomically.
        dry_run: Report what would be written without writing it.
        quiet: Suppress progress output.
    """
    try:
        # Resolve the input path before reading, so writing the output over the
        # input is refused rather than destroying it.
        input_name = getattr(input_file, "name", None)
        input_path = Path(input_name) if input_name and input_name != "-" else None
        if output_file is not None:
            validate_file_paths(input_path, output_file, create_dirs=not dry_run)

        lines = _read_input(text, input_file)
        if not lines:
            click.echo("No text to transliterate.", err=True)
            sys.exit(1)

        src, tgt = resolve_pair(source, target, " ".join(lines[:50]))

        if dry_run:
            # Before _run, not after. Downstream of the backend a "dry" run has
            # already downloaded weights and, with --engine llm, already paid
            # for the answer it then throws away. It also used to be ignored
            # entirely without --output, since the check sat inside that branch.
            where = (
                f"to {output_file} as {output_format}" if output_file else "to stdout"
            )
            click.echo(
                f"Would write {len(lines)} result(s) {where}, "
                f"{src}->{tgt} via {','.join(engine)}.",
                err=True,
            )
            return
        # Progress only on an interactive terminal, so it never pollutes
        # captured or redirected stdout (CliRunner mixes stderr into output).
        if not quiet and sys.stderr.isatty():
            click.echo(
                f"Transliterating {len(lines)} line(s) "
                f"{src}->{tgt} via {','.join(engine)}...",
                err=True,
            )

        started = time.time()
        outputs = _run(
            lines,
            source=src,
            target=tgt,
            engine=engine,
            n_best=n_best,
            provider=provider,
            llm_model=llm_model,
            api_key=api_key,
        )
        elapsed = time.time() - started

        if output_file is None:
            click.echo("\n".join(outputs))
            return

        results = [
            TransliterationResult(
                line_number=index + 1,
                input_text=line,
                output_text=out,
                source_lang=src,
                target_lang=tgt,
                confidence=",".join(engine),
                processing_time=elapsed / max(len(lines), 1),
            )
            for index, (line, out) in enumerate(zip(lines, outputs, strict=True))
        ]
        if backup and output_file.exists():
            created = create_backup(output_file)
            if created and not quiet:
                click.echo(f"Backed up to {created}", err=True)
        write_output_safely(results, output_file, output_format, src, tgt, atomic)
        if not quiet:
            click.echo(f"Results written to {output_file}", err=True)

    except BackendsUnavailableError as exc:
        # Without this the CLI printed a blank line and exited 0.
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except UnsupportedPairError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


def _read_input(text: str | None, input_file: IO[str] | None) -> list[str]:
    """Collect input lines from an argument, a file, or stdin.

    Args:
        text: The positional argument, if given.
        input_file: An opened input file, if given.

    Returns:
        Lines to transliterate. Blank lines are preserved so file output lines
        up with file input.
    """
    if text:
        return [text]
    if input_file is not None:
        name = getattr(input_file, "name", None)
        if name and name != "-":
            return read_input_file(Path(name))
        return [line.rstrip("\n") for line in input_file.readlines()]
    if not sys.stdin.isatty():
        # One element per line, as the file path does. Returning the whole
        # stream as a single element handed the backend a token containing a
        # newline, which matches nothing and collapsed every piped line into
        # one output line -- or, with a table-only chain, into no output at all.
        content = sys.stdin.read()
        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        return lines if any(line.strip() for line in lines) else []
    click.echo("No input provided. Use --help for usage information.", err=True)
    sys.exit(1)


def _run(
    lines: list[str],
    *,
    source: str,
    target: str,
    engine: tuple[str, ...],
    n_best: int,
    provider: str | None,
    llm_model: str | None,
    api_key: str | None,
) -> list[str]:
    """Transliterate lines in one batched pass, keeping blank lines blank.

    Args:
        lines: Input lines.
        source: Source language.
        target: Target language.
        engine: Backend chain.
        n_best: Candidates per line.
        provider: LLM provider.
        llm_model: LLM model name.
        api_key: LLM API key.

    Returns:
        One output line per input line.
    """
    from .api import transliterate_batch

    kwargs: dict[str, Any] = {
        key: value
        for key, value in (
            ("provider", provider),
            ("model", llm_model),
            ("api_key", api_key),
        )
        if value is not None
    }
    nonempty = [line for line in lines if line.strip()]
    preds = (
        transliterate_batch(
            nonempty, source=source, target=target, engine=engine, n=n_best, **kwargs
        )
        if nonempty
        else []
    )
    # n > 1 yields a candidate list per input; join so the file stays line-based.
    rendered = [" | ".join(p) if isinstance(p, list) else str(p) for p in preds]
    remaining = iter(rendered)
    return [next(remaining) if line.strip() else "" for line in lines]


@cli.command()
def languages() -> None:
    """List the directions this installation can transliterate, and how."""
    table = supported()
    local = {direction for direction in table if direction in PAIRS}
    click.echo(f"{'Direction':<26}{'Backend':<10}Status")
    click.echo("-" * 60)
    unavailable = False
    for direction in sorted(table, key=lambda d: (d not in local, d)):
        source, target = direction
        # status(), not supported(): the useful question is whether it will work
        # right now, not whether the direction is in scope.
        states = status(source, target)
        label = f"{source} -> {target}"
        for backend in table[direction]:
            state = states.get(backend, UNAVAILABLE)
            unavailable |= state == UNAVAILABLE
            click.echo(f"{label:<26}{backend:<10}{state}")
            label = ""
    click.echo()
    click.echo(f"{len(local)} direction(s) can run locally.")
    if unavailable:
        click.echo(
            "Some backends are unavailable. A lookup table is built with:\n"
            "  python training/build_lookup.py --lang <language>"
        )
    click.echo(f"Languages known: {', '.join(sorted(LANGUAGES))}")


@cli.command()
def info() -> None:
    """Show information about the indicate package."""
    from .resources import HF_REPO, HF_REVISION
    from .transliterator import EMBEDDING_DIM, UNITS, model_for

    click.echo("Indicate - Indic transliteration")
    click.echo("=" * 45)
    click.echo()
    click.echo("Local model: encoder-decoder with Luong attention")
    click.echo(f"Embedding dimension: {EMBEDDING_DIM}")
    click.echo(f"LSTM units: {UNITS}")
    click.echo()
    click.echo("Training data sources:")
    for name in (
        "ESPN Cricinfo",
        "Election affidavits",
        "Google Dakshina",
        "IIT Bombay corpus",
    ):
        click.echo(f"  - {name}")
    click.echo()
    click.echo(f"Weights: {HF_REPO}@{HF_REVISION} (Hugging Face; cached on first use)")
    for pair in PAIRS.values():
        present = Path(model_for(pair).weights_dir).exists()
        state = "local copy present" if present else "downloaded on first use"
        click.echo(f"  {pair.source} -> {pair.target}: {state}")
    click.echo()
    click.echo("Backends: " + ", ".join(sorted(KNOWN)))
    click.echo("Run 'indicate languages' for supported directions.")


def main() -> None:
    """Entry point for the ``indicate`` console script."""
    cli()


if __name__ == "__main__":
    main()
