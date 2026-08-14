"""Safe file handling utilities for transliteration operations."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from .logging import get_logger

logger = get_logger()


class OutputFormat:
    """Output format definitions and handlers."""

    TEXT = "text"
    JSON = "json"

    VALID_FORMATS: ClassVar[list[str]] = [TEXT, JSON]


class TransliterationResult:
    """Represents a single transliteration result with metadata."""

    def __init__(
        self,
        line_number: int,
        input_text: str,
        output_text: str,
        source_lang: str,
        target_lang: str,
        confidence: str = "unknown",
        error: str | None = None,
        processing_time: float | None = None,
    ):
        self.line_number = line_number
        self.input_text = input_text
        self.output_text = output_text
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.confidence = confidence
        self.error = error
        self.processing_time = processing_time
        self.timestamp = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert transliteration result to dictionary format.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "line_number": self.line_number,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "confidence": self.confidence,
            "error": self.error,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp,
        }


class BatchProgress:
    """Tracks progress of batch transliteration operations."""

    def __init__(self, total_lines: int, output_path: Path):
        self.total_lines = total_lines
        self.processed_lines = 0
        self.successful_lines = 0
        self.failed_lines = 0
        self.output_path = output_path
        self.start_time = time.time()
        self.results: list[TransliterationResult] = []

        # Progress file for resuming
        self.progress_file = output_path.with_suffix(".progress.json")

    def add_result(self, result: TransliterationResult):
        """Add a transliteration result."""
        self.results.append(result)
        self.processed_lines += 1

        if result.error is None:
            self.successful_lines += 1
        else:
            self.failed_lines += 1

    def save_progress(self):
        """Save current progress to disk for recovery."""
        progress_data = {
            "total_lines": self.total_lines,
            "processed_lines": self.processed_lines,
            "successful_lines": self.successful_lines,
            "failed_lines": self.failed_lines,
            "start_time": self.start_time,
            "last_update": time.time(),
            "output_path": str(self.output_path),
            "results": [result.to_dict() for result in self.results],
        }

        # Use atomic write for progress file too
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, dir=self.progress_file.parent, suffix=".tmp"
        ) as tmp_file:
            json.dump(progress_data, tmp_file, indent=2, ensure_ascii=False)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        # Atomic rename
        os.rename(tmp_file.name, self.progress_file)

    @classmethod
    def load_progress(cls, progress_file: Path) -> BatchProgress | None:
        """Load progress from disk."""
        if not progress_file.exists():
            return None

        try:
            with open(progress_file, encoding="utf-8") as f:
                data = json.load(f)

            progress = cls(data["total_lines"], Path(data["output_path"]))
            progress.processed_lines = data["processed_lines"]
            progress.successful_lines = data["successful_lines"]
            progress.failed_lines = data["failed_lines"]
            progress.start_time = data["start_time"]

            # Rebuild results
            for result_data in data["results"]:
                result = TransliterationResult(
                    line_number=result_data["line_number"],
                    input_text=result_data["input_text"],
                    output_text=result_data["output_text"],
                    source_lang=result_data["source_lang"],
                    target_lang=result_data["target_lang"],
                    confidence=result_data.get("confidence", "unknown"),
                    error=result_data.get("error"),
                    processing_time=result_data.get("processing_time"),
                )
                progress.results.append(result)

            return progress

        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
            return None

    def cleanup(self):
        """Remove progress file after successful completion."""
        if self.progress_file.exists():
            self.progress_file.unlink()


def validate_file_paths(
    input_path: Path | None, output_path: Path | None, *, create_dirs: bool = True
) -> None:
    """Validate input and output file paths for safety.

    Args:
        input_path: Input file path.
        output_path: Output file path.
        create_dirs: Create a missing output directory as part of validating it.
            Pass ``False`` for ``--dry-run``, which promises to write nothing
            and must not leave a directory behind to prove it thought about it.

    Raises:
        ValueError: If paths are invalid or dangerous.
    """
    if input_path and output_path:
        # Check if paths are the same (dangerous!)
        try:
            if input_path.resolve() == output_path.resolve():
                raise ValueError(
                    f"Input and output paths cannot be the same file: {input_path}. "
                    "This would destroy your input data!"
                )
        except (OSError, ValueError):
            # If resolution fails, do string comparison
            if str(input_path) == str(output_path):
                raise ValueError(
                    f"Input and output paths appear to be the same: {input_path}"
                ) from None

    if input_path and not input_path.exists():
        raise ValueError(f"Input file does not exist: {input_path}")

    if output_path:
        # Check if output directory is writable
        output_dir = output_path.parent
        if not output_dir.exists():
            if not create_dirs:
                # Nothing more to check: an absent directory cannot be probed
                # for writability, and creating one to find out is the mutation
                # a dry run is promising not to make.
                return
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise ValueError(
                    f"Cannot create output directory {output_dir}: {e}"
                ) from e

        if not os.access(output_dir, os.W_OK):
            raise ValueError(f"Output directory is not writable: {output_dir}")


def create_backup(file_path: Path) -> Path | None:
    """Create a backup of a file before overwriting.

    Args:
        file_path: File to backup.

    Returns:
        Path to backup file, or None if backup wasn't needed.
    """
    if not file_path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f".backup_{timestamp}{file_path.suffix}")

    try:
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.warning(f"Could not create backup: {e}")
        return None


def write_output_safely(
    results: list[TransliterationResult],
    output_path: Path,
    output_format: str,
    source_lang: str,
    target_lang: str,
    atomic: bool = True,
) -> None:
    """Write transliteration results to file safely.

    Args:
        results: List of transliteration results.
        output_path: Path to output file.
        output_format: Output format (text, json, csv).
        source_lang: Source language.
        target_lang: Target language.
        atomic: Whether to use atomic writing.

    Raises:
        ValueError: If ``output_format`` is not a supported format.
    """
    if output_format not in OutputFormat.VALID_FORMATS:
        raise ValueError(f"Invalid output format: {output_format}")

    # Choose write function
    if atomic:
        _write_file_atomic(
            results, output_path, output_format, source_lang, target_lang
        )
    else:
        _write_file_direct(
            results, output_path, output_format, source_lang, target_lang
        )


def _write_file_atomic(
    results: list[TransliterationResult],
    output_path: Path,
    output_format: str,
    source_lang: str,
    target_lang: str,
) -> None:
    """Write file using atomic operation (temp file + rename)."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=output_path.parent,
        suffix=".tmp",
        prefix=f"{output_path.stem}_",
    ) as tmp_file:
        _write_content(tmp_file, results, output_format, source_lang, target_lang)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())

    # Atomic rename
    os.rename(tmp_file.name, output_path)
    logger.info(f"Safely wrote {len(results)} results to {output_path}")


def _write_file_direct(
    results: list[TransliterationResult],
    output_path: Path,
    output_format: str,
    source_lang: str,
    target_lang: str,
) -> None:
    """Write file directly (less safe but faster)."""
    with open(output_path, "w", encoding="utf-8") as f:
        _write_content(f, results, output_format, source_lang, target_lang)

    logger.info(f"Wrote {len(results)} results to {output_path}")


def _write_content(
    file_handle,
    results: list[TransliterationResult],
    output_format: str,
    source_lang: str,
    target_lang: str,
) -> None:
    """Write content in the specified format."""
    if output_format == OutputFormat.TEXT:
        _write_text_format(file_handle, results)
    elif output_format == OutputFormat.JSON:
        _write_json_format(file_handle, results, source_lang, target_lang)
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def _write_text_format(file_handle, results: list[TransliterationResult]) -> None:
    """Write plain text format - just the transliterated output."""
    for result in results:
        if result.error:
            # For errors, write the original text with a comment
            file_handle.write(f"{result.input_text}  # ERROR: {result.error}\n")
        else:
            file_handle.write(f"{result.output_text}\n")


def _write_json_format(
    file_handle,
    results: list[TransliterationResult],
    source_lang: str,
    target_lang: str,
) -> None:
    """Write structured JSON format with metadata and robust UTF-8 handling."""
    successful_results = [r for r in results if r.error is None]
    failed_results = [r for r in results if r.error is not None]

    output_data = {
        "metadata": {
            "source_language": source_lang,
            "target_language": target_lang,
            "timestamp": datetime.now(UTC).isoformat(),
            "total_lines": len(results),
            "successful_lines": len(successful_results),
            "failed_lines": len(failed_results),
            "format_version": "1.0",
            "encoding": "utf-8",
            "description": "Indic language transliteration results "
            "from indicate package",
        },
        "results": [result.to_dict() for result in results],
    }

    # Use ensure_ascii=False for proper UTF-8 output of Indic characters
    json.dump(output_data, file_handle, indent=2, ensure_ascii=False)


def read_input_file(input_path: Path) -> list[str]:
    """Read input file safely, handling both text and JSON formats.

    Args:
        input_path: Path to input file.

    Returns:
        List of text lines to transliterate.
    """
    # Check if it's a JSON file with previous results
    if input_path.suffix.lower() == ".json":
        return _read_json_input(input_path)
    else:
        return _read_text_input(input_path)


def _read_text_input(input_path: Path) -> list[str]:
    """Read plain text input file with robust UTF-8 handling."""
    try:
        # Try UTF-8 first
        with open(input_path, encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            # Fallback to UTF-8 with BOM
            with open(input_path, encoding="utf-8-sig") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            # Last resort - try with error handling
            with open(input_path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                logger.warning(
                    f"Some characters in {input_path} could not be decoded properly"
                )

    # Remove BOM if present and normalize line endings
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0][1:]

    # Strip line endings but preserve empty lines
    return [line.rstrip("\r\n") for line in lines]


def _read_json_input(input_path: Path) -> list[str]:
    """Read JSON input file and extract text to transliterate.

    Supports both:
    1. Simple array: ["text1", "text2", "text3"]
    2. Previous results format: {"results": [...]}
    """
    try:
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle simple array format
        if isinstance(data, list):
            return data

        # Handle structured results format
        if isinstance(data, dict):
            if "results" in data:
                # Extract input_text from previous results
                texts = []
                for result in data["results"]:
                    if isinstance(result, dict) and "input_text" in result:
                        texts.append(result["input_text"])
                    else:
                        texts.append(str(result))
                return texts
            elif "texts" in data:
                # Alternative format
                return list(data["texts"])
            else:
                # Assume the values are the texts
                return list(data.values())

        # Fallback
        return [str(data)]

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {input_path}: {e}")
        raise ValueError(f"Invalid JSON format in {input_path}") from e
    except Exception as e:
        logger.error(f"Error reading JSON file {input_path}: {e}")
        raise


def check_resume_possibility(output_path: Path) -> BatchProgress | None:
    """Check if a previous batch operation can be resumed.

    Args:
        output_path: Output file path.

    Returns:
        BatchProgress object if resume is possible, None otherwise.
    """
    progress_file = output_path.with_suffix(".progress.json")
    return BatchProgress.load_progress(progress_file)
