#!/usr/bin/env python3
"""
Test file I/O operations for the indicate package.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pytest
from click.testing import CliRunner

from indicate.cli import cli


@pytest.mark.needs_weights
class TestFileIO(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_input_file_basic(self):
        """Test basic input file functionality."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("हिंदी")
            temp_file = f.name

        try:
            result = self.runner.invoke(
                cli, ["transliterate", "--from", "hindi", "--input", temp_file]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("hindi", result.output.lower())
        finally:
            Path(temp_file).unlink()

    def test_output_file_basic(self):
        """Test basic output file functionality."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            temp_output = f.name

        try:
            result = self.runner.invoke(
                cli, ["transliterate", "हिंदी", "--output", temp_output]
            )
            self.assertEqual(result.exit_code, 0)

            with open(temp_output, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("hindi", content.lower())
        finally:
            Path(temp_output).unlink()

    def test_input_output_together(self):
        """Test input and output files together."""
        # Create input file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("गौरव सूद\nराजशेखर चिंतालपति")
            temp_input = f.name

        # Create output file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            temp_output = f.name

        try:
            result = self.runner.invoke(
                cli,
                [
                    "transliterate",
                    "--input",
                    temp_input,
                    "--output",
                    temp_output,  # Use batch mode for multiline files
                ],
            )
            self.assertEqual(result.exit_code, 0)

            with open(temp_output, encoding="utf-8") as f:
                content = f.read()

            self.assertIn("gaurav", content.lower())
            self.assertIn("rajshekhar", content.lower())
        finally:
            Path(temp_input).unlink()
            Path(temp_output).unlink()

    def test_batch_processing_file(self):
        """Test batch processing with file input."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("हिंदी\nगौरव\nराजशेखर\n")
            temp_file = f.name

        try:
            result = self.runner.invoke(
                cli, ["transliterate", "--from", "hindi", "--input", temp_file]
            )
            self.assertEqual(result.exit_code, 0)

            lines = result.output.strip().split("\n")
            self.assertEqual(len(lines), 3)
            self.assertIn("hindi", lines[0].lower())
            self.assertIn("gaurav", lines[1].lower())
            self.assertIn("rajshekhar", lines[2].lower())
        finally:
            Path(temp_file).unlink()

    def test_large_file_processing(self):
        """Test processing of a larger file."""
        lines = ["हिंदी", "गौरव", "राजशेखर", "चिंतालपति", "भारत"] * 10  # 50 lines

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("\n".join(lines))
            temp_file = f.name

        try:
            result = self.runner.invoke(
                cli, ["transliterate", "--from", "hindi", "--input", temp_file]
            )
            self.assertEqual(result.exit_code, 0)

            output_lines = result.output.strip().split("\n")
            self.assertEqual(len(output_lines), 50)
        finally:
            Path(temp_file).unlink()

    def test_empty_input_file(self):
        """Test handling of empty input file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            # Write nothing (empty file)
            temp_file = f.name

        try:
            result = self.runner.invoke(
                cli, ["transliterate", "--from", "hindi", "--input", temp_file]
            )
            self.assertEqual(result.exit_code, 1)
            # Should fail gracefully for empty file with appropriate message
            self.assertIn("No text to transliterate", result.output)
        finally:
            Path(temp_file).unlink()

    def test_nonexistent_input_file(self):
        """Test handling of non-existent input file."""
        result = self.runner.invoke(
            cli,
            ["transliterate", "--from", "hindi", "--input", "/nonexistent/file.txt"],
        )
        self.assertNotEqual(result.exit_code, 0)
        # Should fail gracefully with error message

    def test_invalid_output_path(self):
        """Test handling of invalid output path."""
        result = self.runner.invoke(
            cli,
            ["transliterate", "हिंदी", "--output", "/nonexistent/directory/file.txt"],
        )
        self.assertNotEqual(result.exit_code, 0)
        # Should fail gracefully

    def test_file_encoding_utf8(self):
        """Test proper UTF-8 encoding handling."""
        # Test various Hindi characters
        test_content = "हिंदी भाषा\nदेवनागरी लिपि\nभारतीय संस्कृति"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(test_content)
            temp_file = f.name

        try:
            result = self.runner.invoke(
                cli, ["transliterate", "--from", "hindi", "--input", temp_file]
            )
            self.assertEqual(result.exit_code, 0)

            # Should handle UTF-8 properly
            lines = result.output.strip().split("\n")
            self.assertEqual(len(lines), 3)
        finally:
            Path(temp_file).unlink()

    def test_file_with_mixed_content(self):
        """Test file with mixed Hindi/English content."""
        mixed_content = "हिंदी\nEnglish\n123\nगौरव\n!@#"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(mixed_content)
            temp_file = f.name

        try:
            result = self.runner.invoke(
                cli, ["transliterate", "--from", "hindi", "--input", temp_file]
            )
            self.assertEqual(result.exit_code, 0)

            lines = result.output.strip().split("\n")
            # Some lines may fail to translate, so we expect at least some output
            self.assertGreaterEqual(len(lines), 3)  # At least 3 successful translations
        finally:
            Path(temp_file).unlink()

    def test_file_permissions(self):
        """Test file permission handling."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("हिंदी")
            temp_file = f.name

        try:
            # Make file readable
            Path(temp_file).chmod(0o644)
            result = self.runner.invoke(
                cli, ["transliterate", "--from", "hindi", "--input", temp_file]
            )
            self.assertEqual(result.exit_code, 0)
        finally:
            Path(temp_file).unlink()


if __name__ == "__main__":
    unittest.main()
