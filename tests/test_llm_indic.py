"""
Tests for LLM-based Indic transliteration.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pytest

from indicate.indic_utils import (
    detect_indic_script,
    detect_language_from_script,
    normalize_text_for_transliteration,
    split_mixed_script_text,
    validate_indic_language_pair,
)
from indicate.llm_indic import IndicLLMTransliterator


class TestIndicUtils(unittest.TestCase):
    """Test Indic utility functions."""

    def test_detect_indic_script(self):
        """Test script detection from text."""
        # Hindi/Devanagari
        self.assertEqual(detect_indic_script("नमस्ते"), "devanagari")
        self.assertEqual(detect_indic_script("राजेश कुमार"), "devanagari")

        # Tamil
        self.assertEqual(detect_indic_script("வணக்கம்"), "tamil")
        self.assertEqual(detect_indic_script("சென்னை"), "tamil")

        # Telugu
        self.assertEqual(detect_indic_script("నమస్కారం"), "telugu")

        # Bengali
        self.assertEqual(detect_indic_script("নমস্কার"), "bengali")

        # English/Latin
        self.assertEqual(detect_indic_script("Hello World"), "latin")

        # Mixed (majority wins)
        self.assertEqual(detect_indic_script("नमस्ते Hello"), "devanagari")

        # Empty
        self.assertIsNone(detect_indic_script(""))

    def test_detect_language_from_script(self):
        """Test language detection from script."""
        self.assertEqual(detect_language_from_script("नमस्ते"), "hindi")
        self.assertEqual(detect_language_from_script("வணக்கம்"), "tamil")
        self.assertEqual(detect_language_from_script("Hello"), "english")
        self.assertIsNone(detect_language_from_script(""))

    def test_validate_indic_language_pair(self):
        """Test validation of language pairs."""
        # Valid pairs (at least one Indic)
        self.assertTrue(validate_indic_language_pair("hindi", "english"))
        self.assertTrue(validate_indic_language_pair("english", "tamil"))
        self.assertTrue(validate_indic_language_pair("hindi", "tamil"))
        self.assertTrue(validate_indic_language_pair("bengali", "telugu"))

        # Invalid pair (neither is Indic)
        self.assertFalse(validate_indic_language_pair("english", "latin"))

    def test_normalize_text_for_transliteration(self):
        """Test text normalization."""
        # Remove extra whitespace
        self.assertEqual(normalize_text_for_transliteration("नमस्ते   भाई"), "नमस्ते भाई")

        # Replace special characters
        self.assertEqual(normalize_text_for_transliteration("नमस्ते।"), "नमस्ते.")

        # Rupee symbol
        self.assertEqual(normalize_text_for_transliteration("₹100"), "Rs.100")

    def test_split_mixed_script_text(self):
        """Test splitting mixed script text."""
        segments = split_mixed_script_text("Hello नमस्ते World")
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0], ("Hello", "latin"))
        self.assertEqual(segments[1], ("नमस्ते", "devanagari"))
        self.assertEqual(segments[2], ("World", "latin"))

        # Single script
        segments = split_mixed_script_text("नमस्ते भाई")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0], ("नमस्ते भाई", "devanagari"))


class TestLLMModuleImport(unittest.TestCase):
    """Test LLM module import and basic functionality without API keys."""

    def test_llm_module_import(self):
        """Test that LLM modules can be imported without API keys."""
        # Clear all API keys to simulate no-key environment
        api_keys = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "COHERE_API_KEY",
            "INDICATE_LLM_PROVIDER",
            "INDICATE_LLM_MODEL",
        ]
        original_values = {}
        for key in api_keys:
            if key in os.environ:
                original_values[key] = os.environ[key]
                del os.environ[key]

        try:
            # These imports should work without API keys
            from indicate.indic_utils import detect_language_from_script
            from indicate.llm_indic import IndicLLMTransliterator

            # Basic functionality that doesn't require API calls should work
            detected = detect_language_from_script("नमस्ते")
            self.assertEqual(detected, "hindi")

            # LLM class initialization should fail gracefully with clear error
            with self.assertRaises(ValueError) as context:
                IndicLLMTransliterator("hindi", "english")
            self.assertIn("No LLM provider detected", str(context.exception))

        finally:
            # Restore original environment
            for key, value in original_values.items():
                os.environ[key] = value


class TestIndicLLMTransliterator(unittest.TestCase):
    """Test IndicLLMTransliterator class."""

    def setUp(self):
        """Set up test environment."""
        # Mock environment variables
        os.environ["OPENAI_API_KEY"] = "test-key"

    def tearDown(self):
        """Clean up after tests."""
        # Remove mock environment variables
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    def test_language_normalization(self):
        """Test language input normalization."""
        with patch("indicate.llm_indic.completion"):
            trans = IndicLLMTransliterator("hindi", "english")
            self.assertEqual(trans.source_lang, "hindi")
            self.assertEqual(trans.target_lang, "english")

            # ISO codes
            trans = IndicLLMTransliterator("hi", "en")
            self.assertEqual(trans.source_lang, "hindi")
            self.assertEqual(trans.target_lang, "english")

            # Aliases
            trans = IndicLLMTransliterator("hin", "eng")
            self.assertEqual(trans.source_lang, "hindi")
            self.assertEqual(trans.target_lang, "english")

    def test_invalid_language(self):
        """Test handling of invalid languages."""
        with self.assertRaises(ValueError) as context:
            IndicLLMTransliterator("unknown", "english")
        self.assertIn("Unsupported language", str(context.exception))

    def test_invalid_language_pair(self):
        """Test handling of invalid language pairs."""
        with self.assertRaises(ValueError) as context:
            IndicLLMTransliterator("english", "english")
        self.assertIn(
            "At least one language must be an Indic language", str(context.exception)
        )

    def test_provider_detection(self):
        """Test automatic provider detection."""
        with patch("indicate.llm_indic.completion"):
            # OpenAI
            os.environ["OPENAI_API_KEY"] = "test-openai-key"
            trans = IndicLLMTransliterator("hindi", "english")
            self.assertEqual(trans.provider, "openai")
            del os.environ["OPENAI_API_KEY"]

            # Anthropic
            os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
            trans = IndicLLMTransliterator("hindi", "english")
            self.assertEqual(trans.provider, "anthropic")
            del os.environ["ANTHROPIC_API_KEY"]

            # Google
            os.environ["GOOGLE_API_KEY"] = "test-google-key"
            trans = IndicLLMTransliterator("hindi", "english")
            self.assertEqual(trans.provider, "google")
            del os.environ["GOOGLE_API_KEY"]

    def test_no_provider_detected(self):
        """Test error when no provider is detected."""
        # Clear all API keys
        for key in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        ]:
            if key in os.environ:
                del os.environ[key]

        with self.assertRaises(ValueError) as context:
            IndicLLMTransliterator("hindi", "english")
        self.assertIn("No LLM provider detected", str(context.exception))

    @patch("indicate.llm_indic.completion")
    def test_generate_few_shot_examples(self, mock_completion):
        """Test few-shot example generation."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """
source: नमस्ते
target: Namaste

source: राज
target: Raj

source: भारत
target: Bharat
"""
        mock_completion.return_value = mock_response

        os.environ["OPENAI_API_KEY"] = "test-key"
        # Don't cache examples so we test generation
        trans = IndicLLMTransliterator("hindi", "english", cache_examples=False)
        # Clear any pre-loaded examples
        trans._examples_cache.clear()

        examples = trans.generate_few_shot_examples(3)

        self.assertEqual(len(examples), 3)
        self.assertEqual(examples[0]["source"], "नमस्ते")
        self.assertEqual(examples[0]["target"], "Namaste")
        self.assertEqual(examples[1]["source"], "राज")
        self.assertEqual(examples[1]["target"], "Raj")

    @patch("indicate.llm_indic.completion")
    def test_transliterate(self, mock_completion):
        """Test basic transliteration."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Rajesh Kumar"
        mock_completion.return_value = mock_response

        os.environ["OPENAI_API_KEY"] = "test-key"
        trans = IndicLLMTransliterator("hindi", "english")

        result = trans.transliterate("राजेश कुमार", use_few_shot=False)
        self.assertEqual(result, "Rajesh Kumar")

        # Check that completion was called
        mock_completion.assert_called()
        call_args = mock_completion.call_args
        self.assertEqual(call_args[1]["model"], "gpt-5.4-mini")
        self.assertEqual(len(call_args[1]["messages"]), 2)

    @patch("indicate.llm_indic.completion")
    def test_transliterate_empty(self, mock_completion):
        """Test transliteration of empty text."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        trans = IndicLLMTransliterator("hindi", "english")

        # Empty string should return empty
        result = trans.transliterate("")
        self.assertEqual(result, "")

        # Whitespace should return empty
        result = trans.transliterate("   ")
        self.assertEqual(result, "")

        # Completion should not be called for empty input
        mock_completion.assert_not_called()

    @patch("indicate.llm_indic.completion")
    def test_transliterate_batch(self, mock_completion):
        """Test batch transliteration."""
        # Mock batch response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """1. Rajesh
2. Kumar
3. Delhi"""
        mock_completion.return_value = mock_response

        os.environ["OPENAI_API_KEY"] = "test-key"
        trans = IndicLLMTransliterator("hindi", "english")

        texts = ["राजेश", "कुमार", "दिल्ली"]
        results = trans.transliterate_batch(texts, batch_size=3, use_few_shot=False)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "Rajesh")
        self.assertEqual(results[1], "Kumar")
        self.assertEqual(results[2], "Delhi")

    @patch("indicate.llm_indic.completion")
    def test_transliterate_with_few_shot(self, mock_completion):
        """Test transliteration with few-shot examples."""
        # Single call for transliteration (examples come from pre-built)
        trans_response = MagicMock()
        trans_response.choices[0].message.content = "Gaurav Sood"

        mock_completion.return_value = trans_response

        os.environ["OPENAI_API_KEY"] = "test-key"
        trans = IndicLLMTransliterator("hindi", "english")

        result = trans.transliterate("गौरव सूद", use_few_shot=True)
        self.assertEqual(result, "Gaurav Sood")

        # Check that completion was called once (pre-built examples used)
        self.assertEqual(mock_completion.call_count, 1)

    def test_load_prebuilt_examples(self):
        """Test loading pre-built examples from JSON."""
        # Create mock JSON data
        mock_json_data = {
            "hindi_english": [
                {"source": "नमस्ते", "target": "Namaste"},
                {"source": "राज", "target": "Raj"},
            ]
        }

        with patch("importlib.resources.files") as mock_files:
            # Mock the Traversable returned by files(...) / "llm_examples.json"
            mock_path = MagicMock()
            mock_path.is_file.return_value = True
            mock_path.open = mock_open(read_data=json.dumps(mock_json_data))
            mock_files.return_value.__truediv__.return_value = mock_path

            os.environ["OPENAI_API_KEY"] = "test-key"
            trans = IndicLLMTransliterator("hindi", "english")

            # Check that examples were loaded
            cache_key = ("hindi", "english")
            self.assertIn(cache_key, trans._examples_cache)
            self.assertEqual(len(trans._examples_cache[cache_key]), 2)

    @patch("indicate.llm_indic.completion")
    def test_fallback_examples(self, mock_completion):
        """Test fallback examples when generation fails."""
        # Make completion raise an exception
        mock_completion.side_effect = Exception("API error")

        os.environ["OPENAI_API_KEY"] = "test-key"
        trans = IndicLLMTransliterator("hindi", "english")

        # Should return fallback examples
        examples = trans.generate_few_shot_examples()
        self.assertGreater(len(examples), 0)
        self.assertIn("source", examples[0])
        self.assertIn("target", examples[0])

    @patch("indicate.llm_indic.completion")
    def test_error_handling_in_transliterate(self, mock_completion):
        """Test error handling in transliteration."""
        # Make completion raise an exception
        mock_completion.side_effect = Exception("API error")

        os.environ["OPENAI_API_KEY"] = "test-key"
        trans = IndicLLMTransliterator("hindi", "english")

        with self.assertRaises(RuntimeError) as context:
            trans.transliterate("नमस्ते", use_few_shot=False)
        self.assertIn("Failed to transliterate", str(context.exception))


class TestCLIIntegration(unittest.TestCase):
    """The LLM is reached through ``transliterate --engine llm``.

    There is no longer a separate ``llm`` command: the backend is an argument,
    which is what lets the table intercept it with ``--engine lookup,llm``.
    """

    @patch("indicate.llm_indic.IndicLLMTransliterator")
    def test_the_llm_backend_is_reached_through_the_engine_flag(self, mock_class):
        from click.testing import CliRunner

        from indicate.cli import cli

        client = MagicMock()
        client.transliterate_batch.return_value = ["namaste"]
        mock_class.return_value = client

        runner = CliRunner()
        result = runner.invoke(
            cli, ["transliterate", "नमस्ते", "--from", "hindi", "--engine", "llm"]
        )

        self.assertEqual(result.exit_code, 0, result.output)
        mock_class.assert_called_once()
        # The backend passes the pair positionally, from the registry.
        self.assertEqual(mock_class.call_args[0][:2], ("hindi", "english"))
        client.transliterate_batch.assert_called_once()
        self.assertEqual(client.transliterate_batch.call_args[0][0], ["नमस्ते"])
        self.assertIn("namaste", result.output)

    @patch("indicate.llm_indic.IndicLLMTransliterator")
    def test_the_language_is_auto_detected_for_the_llm_too(self, mock_class):
        from click.testing import CliRunner

        from indicate.cli import cli

        client = MagicMock()
        client.transliterate_batch.return_value = ["namaste"]
        mock_class.return_value = client

        runner = CliRunner()
        result = runner.invoke(cli, ["transliterate", "नमस्ते", "--engine", "llm"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(mock_class.call_args[0][:2], ("hindi", "english"))

    @patch("indicate.llm_indic.IndicLLMTransliterator")
    @pytest.mark.needs_weights
    def test_a_failing_provider_declines_instead_of_raising(self, mock_class):
        # Declining is what lets ("lookup", "llm", "model") degrade to local
        # decoding rather than losing the word.
        from click.testing import CliRunner

        from indicate.cli import cli

        client = MagicMock()
        client.transliterate_batch.side_effect = RuntimeError("no api key")
        mock_class.return_value = client

        runner = CliRunner()
        result = runner.invoke(
            cli, ["transliterate", "नमस्ते", "--from", "hindi", "--engine", "llm,model"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("namaste", result.output)


if __name__ == "__main__":
    unittest.main()
