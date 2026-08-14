#!/usr/bin/env python3
"""
Test CLI commands for the indicate package.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pytest
from click.testing import CliRunner

from indicate.cli import cli


@pytest.mark.needs_weights
class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_cli_version(self):
        """Test that version command works."""
        result = self.runner.invoke(cli, ["--version"])
        self.assertEqual(result.exit_code, 0)
        # Version should be present, but due to metadata caching may not be
        # 0.4.0 immediately
        self.assertTrue(any(c.isdigit() for c in result.output))

    def test_transliterate_basic(self):
        """Test basic hindi2english command with text argument."""
        result = self.runner.invoke(cli, ["transliterate", "हिंदी"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hindi", result.output.lower())

    def test_transliterate_stdin(self):
        """Test hindi2english with stdin input."""
        result = self.runner.invoke(cli, ["transliterate"], input="गौरव")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("gaurav", result.output.lower())

    def test_transliterate_file_input(self):
        """Test hindi2english with file input."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("राजशेखर\n")
            temp_file = f.name

        try:
            result = self.runner.invoke(
                cli, ["transliterate", "--from", "hindi", "--input", temp_file]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("rajshekhar", result.output.lower())
        finally:
            Path(temp_file).unlink()

    def test_transliterate_file_output(self):
        """Test hindi2english with file output."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as outfile:
            temp_output = outfile.name

        try:
            result = self.runner.invoke(
                cli, ["transliterate", "हिंदी", "--output", temp_output]
            )
            self.assertEqual(result.exit_code, 0)

            with open(temp_output, encoding="utf-8") as f:
                output_content = f.read()
            self.assertIn("hindi", output_content.lower())
        finally:
            Path(temp_output).unlink()

    def test_transliterate_batch_mode(self):
        """Test hindi2english with batch processing."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("हिंदी\nगौरव\n")
            temp_file = f.name

        try:
            result = self.runner.invoke(
                cli, ["transliterate", "--from", "hindi", "--input", temp_file]
            )
            self.assertEqual(result.exit_code, 0)
            output_lines = result.output.strip().split("\n")
            self.assertEqual(len(output_lines), 2)
            self.assertIn("hindi", output_lines[0].lower())
            self.assertIn("gaurav", output_lines[1].lower())
        finally:
            Path(temp_file).unlink()

    def test_transliterate_quiet_mode(self):
        """Test hindi2english with quiet mode."""
        result = self.runner.invoke(cli, ["transliterate", "हिंदी", "--quiet"])
        self.assertEqual(result.exit_code, 0)
        # In quiet mode, should have minimal output
        lines = [line for line in result.output.strip().split("\n") if line.strip()]
        self.assertEqual(len(lines), 1)  # Just the translation result

    def test_info_command(self):
        """Test the info command."""
        result = self.runner.invoke(cli, ["info"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Indicate", result.output)
        self.assertIn("encoder-decoder with Luong attention", result.output)

    def test_transliterate_help(self):
        """Test that help command works."""
        result = self.runner.invoke(cli, ["transliterate", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Transliterate TEXT, a file, or standard input", result.output)


if __name__ == "__main__":
    unittest.main()
