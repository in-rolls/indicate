"""
Tests for file safety and handling features.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from indicate.file_utils import (
    BatchProgress,
    OutputFormat,
    TransliterationResult,
    _read_json_input,
    _read_text_input,
    create_backup,
    read_input_file,
    validate_file_paths,
    write_output_safely,
)


class TestFilePathValidation(unittest.TestCase):
    """Test file path validation and safety checks."""

    def test_same_input_output_paths(self):
        """Test detection of same input/output paths."""
        with tempfile.NamedTemporaryFile() as tmp:
            tmp_path = Path(tmp.name)

            # Same exact paths should raise error
            with self.assertRaises(ValueError) as context:
                validate_file_paths(tmp_path, tmp_path)
            error_msg = str(context.exception).lower()
            # Check for either error message variant
            self.assertTrue(
                "cannot be the same file" in error_msg
                or "appear to be the same" in error_msg
            )

    def test_nonexistent_input_file(self):
        """Test error when input file doesn't exist."""
        nonexistent = Path("/nonexistent/file.txt")
        output_path = Path("/tmp/output.txt")

        with self.assertRaises(ValueError) as context:
            validate_file_paths(nonexistent, output_path)
        self.assertIn("does not exist", str(context.exception))

    def test_valid_different_paths(self):
        """Test validation passes for different valid paths."""
        with (
            tempfile.NamedTemporaryFile() as input_file,
            tempfile.NamedTemporaryFile() as output_file,
        ):
            input_path = Path(input_file.name)
            output_path = Path(output_file.name)

            # Should not raise any exception
            validate_file_paths(input_path, output_path)

    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_dir = Path(tmp_dir) / "new" / "subdir"
            output_path = new_dir / "output.txt"

            # Directory doesn't exist yet
            self.assertFalse(new_dir.exists())

            # Validation should create it
            validate_file_paths(None, output_path)
            self.assertTrue(new_dir.exists())


class TestBackupFunctionality(unittest.TestCase):
    """Test backup creation and management."""

    def test_create_backup_existing_file(self):
        """Test backup creation for existing file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            tmp.write("Original content")
            tmp.flush()
            file_path = Path(tmp.name)

        try:
            # Create backup
            backup_path = create_backup(file_path)

            # Backup should exist and have different name
            self.assertIsNotNone(backup_path)
            self.assertTrue(backup_path.exists())
            self.assertNotEqual(file_path, backup_path)
            self.assertIn("backup_", backup_path.name)

            # Content should be preserved
            with Path(backup_path).open() as f:
                backup_content = f.read()
            self.assertEqual(backup_content, "Original content")

        finally:
            # Cleanup
            if file_path.exists():
                file_path.unlink()
            if backup_path and backup_path.exists():
                backup_path.unlink()

    def test_backup_nonexistent_file(self):
        """Test backup of nonexistent file returns None."""
        nonexistent = Path("/nonexistent/file.txt")
        backup_path = create_backup(nonexistent)
        self.assertIsNone(backup_path)


class TestTransliterationResult(unittest.TestCase):
    """Test TransliterationResult class."""

    def test_result_creation(self):
        """Test creating a transliteration result."""
        result = TransliterationResult(
            line_number=1,
            input_text="राज",
            output_text="Raj",
            source_lang="hindi",
            target_lang="english",
            confidence="high",
            processing_time=0.5,
        )

        self.assertEqual(result.line_number, 1)
        self.assertEqual(result.input_text, "राज")
        self.assertEqual(result.output_text, "Raj")
        self.assertEqual(result.source_lang, "hindi")
        self.assertEqual(result.target_lang, "english")
        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.processing_time, 0.5)
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.timestamp)

    def test_result_with_error(self):
        """Test creating result with error."""
        result = TransliterationResult(
            line_number=2,
            input_text="test",
            output_text="",
            source_lang="hindi",
            target_lang="english",
            error="API timeout",
        )

        self.assertEqual(result.error, "API timeout")
        self.assertEqual(result.output_text, "")

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = TransliterationResult(
            line_number=1,
            input_text="नमस्ते",
            output_text="Namaste",
            source_lang="hindi",
            target_lang="english",
            confidence="high",
        )

        result_dict = result.to_dict()

        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["line_number"], 1)
        self.assertEqual(result_dict["input_text"], "नमस्ते")
        self.assertEqual(result_dict["output_text"], "Namaste")
        self.assertEqual(result_dict["source_lang"], "hindi")
        self.assertEqual(result_dict["target_lang"], "english")
        self.assertIn("timestamp", result_dict)


class TestJSONIO(unittest.TestCase):
    """Test JSON input/output handling."""

    def test_json_output_format(self):
        """Test JSON output format structure."""
        results = [
            TransliterationResult(
                line_number=1,
                input_text="राज",
                output_text="Raj",
                source_lang="hindi",
                target_lang="english",
                confidence="high",
                processing_time=0.3,
            ),
            TransliterationResult(
                line_number=2,
                input_text="गौरव",
                output_text="Gaurav",
                source_lang="hindi",
                target_lang="english",
                confidence="high",
                processing_time=0.4,
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            output_path = Path(tmp.name)

        try:
            # Write JSON output
            write_output_safely(
                results, output_path, OutputFormat.JSON, "hindi", "english"
            )

            # Read and validate
            with Path(output_path).open(encoding="utf-8") as f:
                data = json.load(f)

            # Check structure
            self.assertIn("metadata", data)
            self.assertIn("results", data)

            # Check metadata
            metadata = data["metadata"]
            self.assertEqual(metadata["source_language"], "hindi")
            self.assertEqual(metadata["target_language"], "english")
            self.assertEqual(metadata["total_lines"], 2)
            self.assertEqual(metadata["successful_lines"], 2)
            self.assertEqual(metadata["failed_lines"], 0)
            self.assertEqual(metadata["format_version"], "1.0")
            self.assertEqual(metadata["encoding"], "utf-8")

            # Check results
            results_data = data["results"]
            self.assertEqual(len(results_data), 2)
            self.assertEqual(results_data[0]["input_text"], "राज")
            self.assertEqual(results_data[0]["output_text"], "Raj")
            self.assertEqual(results_data[1]["input_text"], "गौरव")
            self.assertEqual(results_data[1]["output_text"], "Gaurav")

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_read_json_input_simple_array(self):
        """Test reading JSON input as simple array."""
        json_data = ["राज", "गौरव", "नमस्ते"]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            json.dump(json_data, tmp, ensure_ascii=False)
            input_path = Path(tmp.name)

        try:
            texts = _read_json_input(input_path)
            self.assertEqual(texts, ["राज", "गौरव", "नमस्ते"])
        finally:
            if input_path.exists():
                input_path.unlink()

    def test_read_json_input_results_format(self):
        """Test reading JSON input from previous results."""
        json_data = {
            "metadata": {"source_language": "hindi"},
            "results": [
                {"input_text": "राज", "output_text": "Raj"},
                {"input_text": "गौरव", "output_text": "Gaurav"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            json.dump(json_data, tmp, ensure_ascii=False)
            input_path = Path(tmp.name)

        try:
            texts = _read_json_input(input_path)
            self.assertEqual(texts, ["राज", "गौरव"])
        finally:
            if input_path.exists():
                input_path.unlink()


class TestTextIO(unittest.TestCase):
    """Test text file input/output handling."""

    def test_read_text_input_utf8(self):
        """Test reading text input with UTF-8 encoding."""
        text_lines = ["राजेश कुमार\n", "गौरव सूद\n", "नमस्ते\n"]

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.writelines(text_lines)
            input_path = Path(tmp.name)

        try:
            lines = _read_text_input(input_path)
            expected = ["राजेश कुमार", "गौरव सूद", "नमस्ते"]
            self.assertEqual(lines, expected)
        finally:
            if input_path.exists():
                input_path.unlink()

    def test_read_text_input_with_bom(self):
        """Test reading text input with BOM."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, encoding="utf-8-sig"
        ) as tmp:
            tmp.write("राजेश\nगौरव")
            input_path = Path(tmp.name)

        try:
            lines = _read_text_input(input_path)
            self.assertEqual(lines, ["राजेश", "गौरव"])
            # Should not start with BOM
            self.assertFalse(lines[0].startswith("\ufeff"))
        finally:
            if input_path.exists():
                input_path.unlink()

    def test_text_output_format(self):
        """Test text output format."""
        results = [
            TransliterationResult(1, "राज", "Raj", "hindi", "english"),
            TransliterationResult(2, "गौरव", "Gaurav", "hindi", "english"),
            TransliterationResult(
                3, "error_text", "", "hindi", "english", error="API error"
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            output_path = Path(tmp.name)

        try:
            write_output_safely(
                results, output_path, OutputFormat.TEXT, "hindi", "english"
            )

            with Path(output_path).open(encoding="utf-8") as f:
                content = f.read()

            lines = content.strip().split("\n")
            self.assertEqual(lines[0], "Raj")
            self.assertEqual(lines[1], "Gaurav")
            self.assertIn("error_text", lines[2])
            self.assertIn("ERROR", lines[2])

        finally:
            if output_path.exists():
                output_path.unlink()


class TestAtomicWriting(unittest.TestCase):
    """Test atomic writing functionality."""

    def test_atomic_write_success(self):
        """Test successful atomic write with real files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_output.json"
            results = [TransliterationResult(1, "राज", "Raj", "hindi", "english")]

            # Should not raise any exception
            write_output_safely(
                results, output_path, OutputFormat.JSON, "hindi", "english", atomic=True
            )

            # Verify file was created and contains expected content
            self.assertTrue(output_path.exists())
            with Path(output_path).open(encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(len(data["results"]), 1)
            self.assertEqual(data["results"][0]["input_text"], "राज")
            self.assertEqual(data["results"][0]["output_text"], "Raj")

    @patch("pathlib.Path.open", autospec=True)
    def test_non_atomic_write(self, mock_file):
        """Test non-atomic write."""
        results = [TransliterationResult(1, "राज", "Raj", "hindi", "english")]
        output_path = Path("/tmp/test_output.json")

        write_output_safely(
            results, output_path, OutputFormat.JSON, "hindi", "english", atomic=False
        )

        # Verify the file was opened directly, at the requested path. autospec
        # keeps `self` in the call record, so the path stays assertable.
        mock_file.assert_called_once_with(output_path, "w", encoding="utf-8")


class TestBatchProgress(unittest.TestCase):
    """Test batch progress tracking and resume functionality."""

    def test_progress_tracking(self):
        """Test basic progress tracking."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            output_path = Path(tmp.name)

            progress = BatchProgress(10, output_path)
            self.assertEqual(progress.total_lines, 10)
            self.assertEqual(progress.processed_lines, 0)

            # Add successful result
            result = TransliterationResult(1, "राज", "Raj", "hindi", "english")
            progress.add_result(result)

            self.assertEqual(progress.processed_lines, 1)
            self.assertEqual(progress.successful_lines, 1)
            self.assertEqual(progress.failed_lines, 0)

            # Add failed result
            error_result = TransliterationResult(
                2, "test", "", "hindi", "english", error="API error"
            )
            progress.add_result(error_result)

            self.assertEqual(progress.processed_lines, 2)
            self.assertEqual(progress.successful_lines, 1)
            self.assertEqual(progress.failed_lines, 1)

    def test_progress_save_load(self):
        """Test saving and loading progress."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output.txt"
            progress_path = output_path.with_suffix(".progress.json")

            # Create progress
            progress = BatchProgress(5, output_path)
            result = TransliterationResult(1, "राज", "Raj", "hindi", "english")
            progress.add_result(result)

            # Save progress
            progress.save_progress()
            self.assertTrue(progress_path.exists())

            # Load progress
            loaded_progress = BatchProgress.load_progress(progress_path)
            self.assertIsNotNone(loaded_progress)
            self.assertEqual(loaded_progress.total_lines, 5)
            self.assertEqual(loaded_progress.processed_lines, 1)
            self.assertEqual(len(loaded_progress.results), 1)
            self.assertEqual(loaded_progress.results[0].input_text, "राज")

    def test_progress_cleanup(self):
        """Test progress file cleanup."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output.txt"
            progress_path = output_path.with_suffix(".progress.json")

            progress = BatchProgress(5, output_path)
            progress.save_progress()
            self.assertTrue(progress_path.exists())

            # Cleanup should remove progress file
            progress.cleanup()
            self.assertFalse(progress_path.exists())


class TestErrorHandling(unittest.TestCase):
    """Test error handling in file operations."""

    def test_invalid_json_input(self):
        """Test handling of invalid JSON input."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            tmp.write("{invalid json")
            input_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as context:
                _read_json_input(input_path)
            self.assertIn("Invalid JSON format", str(context.exception))
        finally:
            if input_path.exists():
                input_path.unlink()

    def test_invalid_output_format(self):
        """Test error on invalid output format."""
        results = [TransliterationResult(1, "test", "test", "hindi", "english")]
        output_path = Path("/tmp/test.txt")

        with self.assertRaises(ValueError) as context:
            write_output_safely(
                results, output_path, "invalid_format", "hindi", "english"
            )
        self.assertIn("Invalid output format", str(context.exception))


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios."""

    def test_complete_workflow_with_backup(self):
        """Test complete workflow: read → process → backup → write."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.txt"
            output_path = Path(tmp_dir) / "output.json"

            # Create input file
            with Path(input_path).open("w", encoding="utf-8") as f:
                f.write("राज\nगौरव\nनमस्ते\n")

            # Create existing output file (to test backup)
            with Path(output_path).open("w", encoding="utf-8") as f:
                f.write("old content")

            # Validate paths (should work)
            validate_file_paths(input_path, output_path)

            # Create backup
            backup_path = create_backup(output_path)
            self.assertIsNotNone(backup_path)
            self.assertTrue(backup_path.exists())

            # Read input
            lines = read_input_file(input_path)
            self.assertEqual(lines, ["राज", "गौरव", "नमस्ते"])

            # Create mock results
            results = [
                TransliterationResult(1, "राज", "Raj", "hindi", "english"),
                TransliterationResult(2, "गौरव", "Gaurav", "hindi", "english"),
                TransliterationResult(3, "नमस्ते", "Namaste", "hindi", "english"),
            ]

            # Write output safely
            write_output_safely(
                results, output_path, OutputFormat.JSON, "hindi", "english"
            )

            # Verify output exists and is valid JSON
            self.assertTrue(output_path.exists())
            with Path(output_path).open(encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(len(data["results"]), 3)
            self.assertEqual(data["results"][0]["output_text"], "Raj")

            # Verify backup still exists with old content
            with Path(backup_path).open(encoding="utf-8") as f:
                backup_content = f.read()
            self.assertEqual(backup_content, "old content")


if __name__ == "__main__":
    unittest.main()
