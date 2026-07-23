#!/usr/bin/env python3
"""Modern CLI for the indicate package using Click."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from typing import IO, cast

import click

from .file_utils import (
    BatchProgress,
    TransliterationResult,
    check_resume_possibility,
    create_backup,
    read_input_file,
    validate_file_paths,
    write_output_safely,
)
from .hindi2english import HindiToEnglish
from .indic_utils import detect_language_from_script
from .llm_indic import IndicLLMTransliterator
from .punjabi2english import PunjabiToEnglish
from .transliterator import Seq2SeqTransliterator


def _get_version() -> str:
    """Get package version from metadata."""
    try:
        return metadata.version("indicate")
    except Exception:
        return "unknown"


@click.group()
@click.version_option(version=_get_version())
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Indicate: Transliterations to/from Indian languages.

    A Python package for transliterating Hindi text to English using
    a TensorFlow-based encoder-decoder model with attention mechanism.
    """
    ctx.ensure_object(dict)


@cli.command()
@click.argument("text", required=False)
@click.option(
    "--input",
    "-i",
    "input_file",
    type=click.File("r", encoding="utf-8"),
    help="Read Hindi text from file instead of argument",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.File("w", encoding="utf-8"),
    help="Write transliterated text to file instead of stdout",
)
@click.option(
    "--batch/--no-batch",
    default=False,
    help="Process input line by line (useful for large files)",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def hindi2english(
    text: str | None,
    input_file: IO[str] | None,
    output_file: IO[str] | None,
    batch: bool,
    quiet: bool,
) -> None:
    """Transliterate Hindi text to English.

    TEXT: Hindi text to transliterate. If not provided, will read from --input or stdin.

    Args:
        text: Hindi (Devanagari) text to transliterate; omit to read from
            ``--input`` or stdin.
        input_file: Read Hindi text from this file instead of the argument.
        output_file: Write the transliteration here instead of stdout.
        batch: Process the input line by line (useful for large files).
        quiet: Suppress progress output.

    Examples:
        # Transliterate text directly
        indicate hindi2english "राजशेखर चिंतालपति"

        # From file
        indicate hindi2english --input hindi.txt --output english.txt

        # From stdin
        echo "गौरव सूद" | indicate hindi2english

        # Batch processing
        indicate hindi2english --input large_file.txt --batch --quiet
    """
    _run_transliterate(HindiToEnglish, text, input_file, output_file, batch, quiet)


@cli.command()
@click.argument("text", required=False)
@click.option(
    "--input",
    "-i",
    "input_file",
    type=click.File("r", encoding="utf-8"),
    help="Read Punjabi (Gurmukhi) text from file instead of argument",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.File("w", encoding="utf-8"),
    help="Write transliterated text to file instead of stdout",
)
@click.option(
    "--batch/--no-batch",
    default=False,
    help="Process input line by line (useful for large files)",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def punjabi2english(
    text: str | None,
    input_file: IO[str] | None,
    output_file: IO[str] | None,
    batch: bool,
    quiet: bool,
) -> None:
    """Transliterate Punjabi (Gurmukhi) text to English.

    TEXT: Punjabi text to transliterate. If not provided, will read from
    --input or stdin.

    Args:
        text: Punjabi (Gurmukhi) text to transliterate; omit to read from
            ``--input`` or stdin.
        input_file: Read Punjabi text from this file instead of the argument.
        output_file: Write the transliteration here instead of stdout.
        batch: Process the input line by line (useful for large files).
        quiet: Suppress progress output.

    Examples:
        # Transliterate text directly
        indicate punjabi2english "ਰਵਿ ਸ਼ਰਮਾ"

        # From file
        indicate punjabi2english --input punjabi.txt --output english.txt
    """
    _run_transliterate(PunjabiToEnglish, text, input_file, output_file, batch, quiet)


def _run_transliterate(
    model: type[Seq2SeqTransliterator],
    text: str | None,
    input_file: IO[str] | None,
    output_file: IO[str] | None,
    batch: bool,
    quiet: bool,
) -> None:
    """Shared body for the per-language transliteration commands."""
    try:
        # Determine input source
        if text:
            input_text = text
        elif input_file:
            if batch:
                # Process the whole file in one batched pass (fast).
                _process_batch(input_file, output_file, quiet, model)
                return
            else:
                input_text = input_file.read().strip()
        else:
            # Read from stdin
            if not sys.stdin.isatty():
                input_text = sys.stdin.read().strip()
            else:
                click.echo(
                    "No input provided. Use --help for usage information.", err=True
                )
                sys.exit(1)

        if not input_text:
            click.echo("No text to transliterate.", err=True)
            sys.exit(1)

        # Transliterate
        if not quiet:
            click.echo("Transliterating...", err=True)

        result = cast("str", model.transliterate(input_text))

        # Output result
        if output_file:
            output_file.write(result)
            if not quiet:
                click.echo(f"Result written to {output_file.name}", err=True)
        else:
            click.echo(result)

    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _process_batch(
    input_file: IO[str],
    output_file: IO[str] | None,
    quiet: bool,
    model: type[Seq2SeqTransliterator],
) -> None:
    """Transliterate a file in one batched pass, preserving blank lines."""
    lines = [line.strip() for line in input_file.readlines()]
    nonempty = [line for line in lines if line]
    # Only show progress on an interactive terminal, so it never pollutes
    # captured/redirected stdout (CliRunner mixes stderr into output).
    if not quiet and sys.stderr.isatty():
        click.echo(f"Transliterating {len(nonempty)} lines...", err=True)

    preds = cast("list[str]", model.transliterate_batch(nonempty)) if nonempty else []
    it = iter(preds)
    results = [next(it) if line else "" for line in lines]

    output_text = "\n".join(results)
    if output_file:
        output_file.write(output_text)
        if not quiet:
            click.echo(f"Results written to {output_file.name}", err=True)
    else:
        click.echo(output_text)


@cli.command()
@click.argument("text", required=False)
@click.option(
    "--source",
    "-s",
    help="Source language (e.g., hindi, tamil, telugu). "
    "Auto-detected if not specified.",
)
@click.option(
    "--target", "-t", default="english", help="Target language (default: english)"
)
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Read text from file instead of argument",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write transliterated text to file instead of stdout",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format: text for stdout, json for files (default: text)",
)
@click.option(
    "--provider",
    "-p",
    help="LLM provider (openai, anthropic, google). "
    "Auto-detected from environment if not specified.",
)
@click.option(
    "--model", "-m", help="Specific model to use (e.g., gpt-4, claude-3-opus)"
)
@click.option(
    "--api-key",
    help="API key for the LLM provider. Prefer setting environment variables instead.",
)
@click.option(
    "--show-examples", is_flag=True, help="Show the few-shot examples being used"
)
@click.option(
    "--no-examples",
    is_flag=True,
    help="Skip few-shot examples (faster but potentially less accurate)",
)
@click.option(
    "--batch/--no-batch",
    default=False,
    help="Process input line by line (useful for large files)",
)
@click.option(
    "--backup", is_flag=True, help="Create backup of output file if it exists"
)
@click.option(
    "--resume", is_flag=True, help="Resume from previous interrupted batch operation"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without making changes",
)
@click.option(
    "--atomic/--no-atomic", default=True, help="Use atomic file writing (default: true)"
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def llm(
    text: str | None,
    source: str | None,
    target: str,
    input_path: Path | None,
    output_path: Path | None,
    output_format: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    show_examples: bool,
    no_examples: bool,
    batch: bool,
    backup: bool,
    resume: bool,
    dry_run: bool,
    atomic: bool,
    quiet: bool,
) -> None:
    """LLM-based transliteration for Indic languages.

    TEXT: Text to transliterate. If not provided, will read from --input or stdin.

    Args:
        text: Text to transliterate; omit to read from ``--input`` or stdin.
        source: Source language (e.g. hindi, tamil); auto-detected if omitted.
        target: Target language (default: english).
        input_path: Read text from this file instead of the argument.
        output_path: Write output to this file instead of stdout.
        output_format: Output format, ``text`` or ``json``.
        provider: LLM provider (openai, anthropic, google); auto-detected from
            the environment if omitted.
        model: Specific model to use; uses the provider default if omitted.
        api_key: API key for the provider; prefer environment variables.
        show_examples: Print the few-shot examples being used, then continue.
        no_examples: Skip few-shot examples (faster, possibly less accurate).
        batch: Process the input line by line (useful for large files).
        backup: Back up the output file first if it already exists.
        resume: Resume a previously interrupted batch operation.
        dry_run: Show what would be processed without writing anything.
        atomic: Use atomic file writing.
        quiet: Suppress progress output.

    Examples:
        # Hindi to English (auto-detected)
        indicate llm "राजशेखर चिंतालपति"

        # Tamil to English
        indicate llm "முருகன்" --source tamil

        # Between Indic languages
        indicate llm "नमस्ते" --source hindi --target tamil

        # From file with JSON output
        indicate llm --input names.txt --output results.json --format json

        # Safe batch processing with backup
        indicate llm --input names.txt --output results.txt --batch --backup --atomic

        # Resume interrupted batch operation
        indicate llm --input names.txt --output results.txt --resume

        # Dry run to preview changes
        indicate llm --input names.txt --dry-run

        # With specific provider/model
        indicate llm "गौरव सूद" --provider openai --model gpt-4

        # Show examples being used
        indicate llm --show-examples --source bengali --target english
    """
    try:
        # Validate file paths early for safety
        if input_path and output_path:
            validate_file_paths(input_path, output_path)

        # Handle resume functionality
        if resume and output_path:
            progress = check_resume_possibility(output_path)
            if progress:
                if not quiet:
                    click.echo(
                        f"Resuming from {progress.processed_lines}"
                        f"/{progress.total_lines} lines",
                        err=True,
                    )
                _resume_batch_processing(
                    progress,
                    source,
                    target,
                    provider,
                    model,
                    api_key,
                    no_examples,
                    output_format,
                    backup,
                    atomic,
                    quiet,
                )
                return
            else:
                click.echo("No previous operation found to resume", err=True)
                sys.exit(1)

        # Determine input source and processing mode
        if text:
            # Single text transliteration
            _process_single_text(
                text,
                source,
                target,
                output_path,
                output_format,
                provider,
                model,
                api_key,
                show_examples,
                no_examples,
                backup,
                atomic,
                quiet,
            )
        elif input_path:
            # File-based processing
            if batch:
                _process_file_batch(
                    input_path,
                    output_path,
                    source,
                    target,
                    provider,
                    model,
                    api_key,
                    no_examples,
                    output_format,
                    backup,
                    dry_run,
                    atomic,
                    quiet,
                )
            else:
                _process_file_single(
                    input_path,
                    output_path,
                    source,
                    target,
                    provider,
                    model,
                    api_key,
                    show_examples,
                    no_examples,
                    output_format,
                    backup,
                    atomic,
                    quiet,
                )
        else:
            # Stdin or just show examples
            if not sys.stdin.isatty():
                input_text = sys.stdin.read().strip()
                _process_single_text(
                    input_text,
                    source,
                    target,
                    output_path,
                    output_format,
                    provider,
                    model,
                    api_key,
                    show_examples,
                    no_examples,
                    backup,
                    atomic,
                    quiet,
                )
            elif show_examples:
                _show_examples_only(
                    source if source else "hindi",
                    target,
                    provider,
                    model,
                    api_key,
                    quiet,
                )
            else:
                click.echo(
                    "No input provided. Use --help for usage information.", err=True
                )
                sys.exit(1)

    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _process_single_text(
    text: str,
    source: str | None,
    target: str,
    output_path: Path | None,
    output_format: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    show_examples: bool,
    no_examples: bool,
    backup: bool,
    atomic: bool,
    quiet: bool,
) -> None:
    """Process a single text input using LLM transliteration.

    Args:
        text: Text to transliterate.
        source: Source language, auto-detected if None.
        target: Target language.
        output_path: Output file path, stdout if None.
        output_format: Output format (text or json).
        provider: LLM provider, auto-detected if None.
        model: LLM model, uses default if None.
        api_key: API key, uses environment if None.
        show_examples: Whether to show few-shot examples.
        no_examples: Whether to skip few-shot examples.
        backup: Whether to create backup of output file.
        atomic: Whether to use atomic file writing.
        quiet: Whether to suppress progress output.
    """
    # Auto-detect source language if not specified
    if not source and text:
        detected = detect_language_from_script(text)
        if detected:
            source = detected
            if not quiet:
                click.echo(f"Auto-detected source language: {source}", err=True)
        else:
            click.echo(
                "Could not auto-detect source language. Please specify with --source",
                err=True,
            )
            sys.exit(1)

    # Default when no --source was given and there was nothing to auto-detect from.
    source = source or "hindi"

    # Initialize transliterator
    if not quiet:
        click.echo("Initializing LLM transliterator...", err=True)

    try:
        transliterator = IndicLLMTransliterator(
            source_lang=source,
            target_lang=target,
            provider=provider,
            model=model,
            api_key=api_key,
        )
    except Exception as e:
        click.echo(f"Failed to initialize transliterator: {e}", err=True)
        click.echo("\nMake sure you have set your API key:", err=True)
        click.echo("  export OPENAI_API_KEY=your-key", err=True)
        click.echo("  export ANTHROPIC_API_KEY=your-key", err=True)
        click.echo("  export GOOGLE_API_KEY=your-key", err=True)
        sys.exit(1)

    # Show examples if requested
    if show_examples:
        examples = transliterator.generate_few_shot_examples()
        click.echo(f"\nFew-shot examples for {source} → {target}:")
        click.echo("-" * 50)
        for ex in examples:
            click.echo(f"{ex['source']} → {ex['target']}")
        click.echo()

    # Transliterate
    if not quiet:
        click.echo("Transliterating...", err=True)

    start_time = time.time()
    result_text = transliterator.transliterate(text, use_few_shot=not no_examples)
    processing_time = time.time() - start_time

    # Create result object
    result = TransliterationResult(
        line_number=1,
        input_text=text,
        output_text=result_text,
        source_lang=source,
        target_lang=target,
        confidence="high",  # Single API call, assume good
        processing_time=processing_time,
    )

    # Output result
    if output_path:
        # Create backup if requested
        if backup and output_path.exists():
            create_backup(output_path)

        # Write to file
        write_output_safely(
            [result], output_path, output_format, source, target, atomic
        )

        if not quiet:
            click.echo(f"Result written to {output_path}", err=True)
    else:
        # Output to stdout (always text format)
        click.echo(result_text)


def _show_examples_only(
    source: str,
    target: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    quiet: bool,
) -> None:
    """Show few-shot examples without processing any text."""
    try:
        transliterator = IndicLLMTransliterator(
            source_lang=source,
            target_lang=target,
            provider=provider,
            model=model,
            api_key=api_key,
        )

        examples = transliterator.generate_few_shot_examples()
        click.echo(f"\nFew-shot examples for {source} → {target}:")
        click.echo("-" * 50)
        for ex in examples:
            click.echo(f"{ex['source']} → {ex['target']}")
        click.echo()

    except Exception as e:
        click.echo(f"Failed to initialize transliterator: {e}", err=True)
        sys.exit(1)


def _process_file_single(
    input_path: Path,
    output_path: Path | None,
    source: str | None,
    target: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    show_examples: bool,
    no_examples: bool,
    output_format: str,
    backup: bool,
    atomic: bool,
    quiet: bool,
) -> None:
    """Process entire file as single text block."""
    # Read input file
    lines = read_input_file(input_path)
    full_text = "\n".join(lines) if lines else ""

    if not full_text.strip():
        click.echo("Input file is empty", err=True)
        return

    # Process as single text
    _process_single_text(
        full_text,
        source,
        target,
        output_path,
        output_format,
        provider,
        model,
        api_key,
        show_examples,
        no_examples,
        backup,
        atomic,
        quiet,
    )


def _process_file_batch(
    input_path: Path,
    output_path: Path | None,
    source: str | None,
    target: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    no_examples: bool,
    output_format: str,
    backup: bool,
    dry_run: bool,
    atomic: bool,
    quiet: bool,
) -> None:
    """Process file line by line in batches."""
    # Read input file
    lines = read_input_file(input_path)

    if not lines:
        click.echo("Input file is empty", err=True)
        return

    # Auto-detect source if not specified
    if not source:
        for line in lines:
            if line.strip():
                detected = detect_language_from_script(line.strip())
                if detected:
                    source = detected
                    if not quiet:
                        click.echo(f"Auto-detected source language: {source}", err=True)
                    break

    if not source:
        click.echo(
            "Could not auto-detect source language. Please specify with --source",
            err=True,
        )
        sys.exit(1)

    # Dry run mode
    if dry_run:
        non_empty_lines = [line for line in lines if line.strip()]
        click.echo(
            f"Would process {len(non_empty_lines)} non-empty lines "
            f"from {len(lines)} total lines"
        )
        click.echo(f"Source language: {source}")
        click.echo(f"Target language: {target}")
        click.echo(f"Output format: {output_format}")
        if output_path:
            click.echo(f"Output file: {output_path}")
        click.echo("\nFirst few lines to process:")
        for i, line in enumerate(non_empty_lines[:5]):
            click.echo(f"  {i + 1}: {line}")
        if len(non_empty_lines) > 5:
            click.echo(f"  ... and {len(non_empty_lines) - 5} more")
        return

    # Initialize transliterator
    if not quiet:
        click.echo("Initializing LLM transliterator...", err=True)

    try:
        transliterator = IndicLLMTransliterator(
            source_lang=source,
            target_lang=target,
            provider=provider,
            model=model,
            api_key=api_key,
        )
    except Exception as e:
        click.echo(f"Failed to initialize transliterator: {e}", err=True)
        click.echo("\nMake sure you have set your API key:", err=True)
        click.echo("  export OPENAI_API_KEY=your-key", err=True)
        click.echo("  export ANTHROPIC_API_KEY=your-key", err=True)
        click.echo("  export GOOGLE_API_KEY=your-key", err=True)
        sys.exit(1)

    # Create progress tracker
    progress = BatchProgress(len(lines), output_path) if output_path else None

    # Create backup if requested
    if backup and output_path and output_path.exists():
        create_backup(output_path)

    # Process lines with progress
    results = []
    lines_iter: Iterable[tuple[int, str]]
    if not quiet:
        from tqdm import tqdm

        lines_iter = tqdm(
            enumerate(lines, 1), total=len(lines), desc="Transliterating", unit="lines"
        )
    else:
        lines_iter = enumerate(lines, 1)

    for line_num, line in lines_iter:
        line = line.strip()

        if line:
            try:
                start_time = time.time()
                output_text = transliterator.transliterate(
                    line, use_few_shot=not no_examples
                )
                processing_time = time.time() - start_time

                result = TransliterationResult(
                    line_number=line_num,
                    input_text=line,
                    output_text=output_text,
                    source_lang=source,
                    target_lang=target,
                    confidence="high",
                    processing_time=processing_time,
                )
            except Exception as e:
                result = TransliterationResult(
                    line_number=line_num,
                    input_text=line,
                    output_text="",
                    source_lang=source,
                    target_lang=target,
                    error=str(e),
                )
        else:
            # Empty line
            result = TransliterationResult(
                line_number=line_num,
                input_text="",
                output_text="",
                source_lang=source,
                target_lang=target,
                confidence="n/a",
            )

        results.append(result)

        # Update progress and save periodically
        if progress:
            progress.add_result(result)
            if line_num % 10 == 0:  # Save progress every 10 lines
                progress.save_progress()

    # Write final output
    if output_path:
        write_output_safely(results, output_path, output_format, source, target, atomic)
        if not quiet:
            successful = sum(1 for r in results if r.error is None)
            click.echo(
                f"Processed {len(results)} lines, {successful} successful", err=True
            )

        # Clean up progress file on success
        if progress:
            progress.cleanup()
    else:
        # Output to stdout (text format only)
        for result in results:
            if result.error:
                click.echo(f"{result.input_text}  # ERROR: {result.error}")
            else:
                click.echo(result.output_text)


def _resume_batch_processing(
    progress: BatchProgress,
    source: str | None,
    target: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    no_examples: bool,
    output_format: str,
    backup: bool,
    atomic: bool,
    quiet: bool,
) -> None:
    """Resume an interrupted batch processing operation."""
    # Re-read input file
    input_path = Path(str(progress.output_path).replace(".progress.json", ""))
    if not input_path.exists():
        click.echo(f"Cannot find original input file: {input_path}", err=True)
        sys.exit(1)

    lines = read_input_file(input_path)

    # Initialize transliterator
    if not quiet:
        click.echo("Resuming batch processing...", err=True)

    try:
        transliterator = IndicLLMTransliterator(
            source_lang=source or progress.results[0].source_lang
            if progress.results
            else "hindi",
            target_lang=target,
            provider=provider,
            model=model,
            api_key=api_key,
        )
    except Exception as e:
        click.echo(f"Failed to initialize transliterator: {e}", err=True)
        sys.exit(1)

    # Process remaining lines
    for line_num in range(progress.processed_lines + 1, len(lines) + 1):
        line = lines[line_num - 1].strip() if line_num <= len(lines) else ""

        if line:
            try:
                start_time = time.time()
                output_text = transliterator.transliterate(
                    line, use_few_shot=not no_examples
                )
                processing_time = time.time() - start_time

                result = TransliterationResult(
                    line_number=line_num,
                    input_text=line,
                    output_text=output_text,
                    source_lang=source or progress.results[0].source_lang,
                    target_lang=target,
                    confidence="high",
                    processing_time=processing_time,
                )
            except Exception as e:
                result = TransliterationResult(
                    line_number=line_num,
                    input_text=line,
                    output_text="",
                    source_lang=source or progress.results[0].source_lang,
                    target_lang=target,
                    error=str(e),
                )
        else:
            result = TransliterationResult(
                line_number=line_num,
                input_text="",
                output_text="",
                source_lang=source or progress.results[0].source_lang,
                target_lang=target,
                confidence="n/a",
            )

        progress.add_result(result)
        progress.save_progress()

        if not quiet and line_num % 10 == 0:
            click.echo(
                f"Processed {progress.processed_lines}/{progress.total_lines} lines",
                err=True,
            )

    # Write final output
    write_output_safely(
        progress.results,
        progress.output_path,
        output_format,
        source or progress.results[0].source_lang,
        target,
        atomic,
    )

    if not quiet:
        click.echo(
            f"Resume completed: {progress.successful_lines} successful, "
            f"{progress.failed_lines} failed",
            err=True,
        )

    # Clean up progress file
    progress.cleanup()


@cli.command()
def info() -> None:
    """Show information about the indicate package."""
    click.echo("Indicate - Hindi to English Transliteration")
    click.echo("=" * 45)
    click.echo()
    click.echo("Model Architecture: Encoder-Decoder with Luong attention")
    click.echo("Embedding dimension: 256")
    click.echo("LSTM units: 1024")
    click.echo()
    click.echo("Training Data Sources:")
    click.echo("  • ESPN Cricinfo")
    click.echo("  • Election affidavits")
    click.echo("  • Google Dakshina dataset")
    click.echo("  • IIT Bombay corpus")
    click.echo()

    # Model weights are hosted on HF and downloaded/cached on first use.
    try:
        from pathlib import Path

        click.echo(
            f"Weights: {HindiToEnglish.HF_REPO}@{HindiToEnglish.HF_REVISION} "
            "(Hugging Face; cached on first use)"
        )
        if Path(HindiToEnglish.get_model_path()).exists():
            click.echo("✓ Local weights present (using local copy)")
        else:
            click.echo("  No local weights — downloaded from HF on first transliterate")
    except Exception as e:
        click.echo(f"⚠ Could not locate model: {e}")

    click.echo()
    click.echo("LLM Support:")
    click.echo("  ✓ Multiple Indic languages supported")
    click.echo("  ✓ Providers: OpenAI, Anthropic, Google, Cohere")
    click.echo("  ✓ Use 'indicate llm --help' for LLM transliteration")


if __name__ == "__main__":
    cli()
