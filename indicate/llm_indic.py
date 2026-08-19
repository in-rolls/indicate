"""LLM-based transliteration for Indic languages using LiteLLM."""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar

from litellm import completion

from .logging import get_logger


def _response_text(response: Any) -> str:
    """Extract the message text from a non-streaming litellm response."""
    return response.choices[0].message.content or ""


logger = get_logger()


class IndicLLMTransliterator:
    """LLM-based transliterator for Indic languages.

    Args:
        source_lang: Source language (e.g., 'hindi', 'tamil')
        target_lang: Target language (e.g., 'english')
        provider: LLM provider (openai, anthropic, etc.). Auto-detected if not
            provided.
        model: Specific model to use. Uses provider defaults if not provided.
        api_key: API key. Uses environment variables if not provided.
        temperature: LLM temperature for consistency (lower = more consistent).
        cache_examples: Whether to cache generated few-shot examples.
    """

    INDIC_LANGUAGES: ClassVar[dict[str, dict[str, str]]] = {
        "hindi": {"native": "हिन्दी", "script": "devanagari", "iso": "hi"},
        "tamil": {"native": "தமிழ்", "script": "tamil", "iso": "ta"},
        "telugu": {"native": "తెలుగు", "script": "telugu", "iso": "te"},
        "bengali": {"native": "বাংলা", "script": "bengali", "iso": "bn"},
        "kannada": {"native": "ಕನ್ನಡ", "script": "kannada", "iso": "kn"},
        "malayalam": {"native": "മലയാളം", "script": "malayalam", "iso": "ml"},
        "gujarati": {"native": "ગુજરાતી", "script": "gujarati", "iso": "gu"},
        "punjabi": {"native": "ਪੰਜਾਬੀ", "script": "gurmukhi", "iso": "pa"},
        "marathi": {"native": "मराठी", "script": "devanagari", "iso": "mr"},
        "odia": {"native": "ଓଡ଼ିଆ", "script": "odia", "iso": "or"},
        "urdu": {"native": "اردو", "script": "arabic", "iso": "ur"},
        "sanskrit": {"native": "संस्कृतम्", "script": "devanagari", "iso": "sa"},
        "english": {"native": "English", "script": "latin", "iso": "en"},
    }

    # Default model preferences by provider
    DEFAULT_MODELS: ClassVar[dict[str, str]] = {
        "openai": "gpt-5.4-mini",
        "anthropic": "claude-3-opus-20240229",
        "google": "gemini-pro",
        "cohere": "command-r-plus",
    }

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        cache_examples: bool = True,
    ):
        self.source_lang = self._normalize_language(source_lang)
        self.target_lang = self._normalize_language(target_lang)
        self._validate_language_pair()

        self.provider = provider or self._detect_provider()
        self.model = model or self._get_default_model()
        self.temperature = temperature
        self.cache_examples = cache_examples

        # Set API key if provided
        if api_key:
            self._set_api_key(api_key)

        # Cache for few-shot examples
        self._examples_cache: dict[tuple[str, str], list[dict[str, str]]] = {}

        # Load pre-built examples if available
        self._load_prebuilt_examples()

        logger.info(
            "Initialized IndicLLMTransliterator: %s → %s, provider=%s, model=%s",
            self.source_lang,
            self.target_lang,
            self.provider,
            self.model,
        )

    def _normalize_language(self, lang: str) -> str:
        """Normalize language input to standard form."""
        lang = lang.lower().strip()

        # Check if it's already a known language
        if lang in self.INDIC_LANGUAGES:
            return lang

        # Check ISO codes
        for key, info in self.INDIC_LANGUAGES.items():
            if info["iso"] == lang:
                return key

        # Common aliases
        aliases = {
            "eng": "english",
            "hin": "hindi",
            "tam": "tamil",
            "tel": "telugu",
            "ben": "bengali",
            "kan": "kannada",
            "mal": "malayalam",
            "guj": "gujarati",
            "pan": "punjabi",
            "pun": "punjabi",
            "mar": "marathi",
            "ori": "odia",
            "odi": "odia",
            "urd": "urdu",
            "san": "sanskrit",
        }

        if lang in aliases:
            return aliases[lang]

        raise ValueError(
            f"Unsupported language: {lang}. "
            f"Supported languages: {', '.join(self.INDIC_LANGUAGES.keys())}"
        )

    def _validate_language_pair(self):
        """Ensure at least one language is Indic."""
        indic_scripts = {
            "devanagari",
            "tamil",
            "telugu",
            "bengali",
            "kannada",
            "malayalam",
            "gujarati",
            "gurmukhi",
            "odia",
            "arabic",
        }

        source_script = self.INDIC_LANGUAGES[self.source_lang]["script"]
        target_script = self.INDIC_LANGUAGES[self.target_lang]["script"]

        if source_script not in indic_scripts and target_script not in indic_scripts:
            raise ValueError(
                "At least one language must be an Indic language. "
                f"Got: {self.source_lang} → {self.target_lang}"
            )

    def _detect_provider(self) -> str:
        """Auto-detect LLM provider from environment variables."""
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            return "google"
        if os.environ.get("COHERE_API_KEY"):
            return "cohere"
        if os.environ.get("INDICATE_LLM_PROVIDER"):
            return os.environ["INDICATE_LLM_PROVIDER"]
        raise ValueError(
            "No LLM provider detected. Please set one of: "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, "
            "or specify provider explicitly."
        )

    def _get_default_model(self) -> str:
        """Get default model for the provider."""
        if self.provider in self.DEFAULT_MODELS:
            return self.DEFAULT_MODELS[self.provider]

        # Check environment variable
        if os.environ.get("INDICATE_LLM_MODEL"):
            return os.environ["INDICATE_LLM_MODEL"]

        # Fallback to provider defaults
        return f"{self.provider}/default"

    def _set_api_key(self, api_key: str):
        """Set API key for the provider."""
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GEMINI_API_KEY",
            "cohere": "COHERE_API_KEY",
        }

        if self.provider in key_mapping:
            os.environ[key_mapping[self.provider]] = api_key

    def _load_prebuilt_examples(self):
        """Load pre-built transliteration examples."""
        try:
            from importlib.resources import files

            examples_path = files("indicate.data") / "llm_examples.json"

            if examples_path.is_file():
                with examples_path.open(encoding="utf-8") as f:
                    examples = json.load(f)

                    key = f"{self.source_lang}_{self.target_lang}"
                    if key in examples:
                        self._examples_cache[(self.source_lang, self.target_lang)] = (
                            examples[key]
                        )
                        logger.info(
                            "Loaded %d pre-built examples for %s",
                            len(examples[key]),
                            key,
                        )
        except Exception as e:
            logger.debug("Could not load pre-built examples: %s", e)

    def generate_few_shot_examples(self, num_examples: int = 5) -> list[dict[str, str]]:
        """Generate few-shot transliteration examples for the language pair.

        Args:
            num_examples: Number of examples to generate.

        Returns:
            List of dictionaries with 'source' and 'target' keys.
        """
        cache_key = (self.source_lang, self.target_lang)

        # Return cached examples if available
        if (
            cache_key in self._examples_cache
            and len(self._examples_cache[cache_key]) >= num_examples
        ):
            return self._examples_cache[cache_key][:num_examples]

        # Generate examples using LLM
        self.INDIC_LANGUAGES[self.source_lang]["native"]
        self.INDIC_LANGUAGES[self.target_lang]["native"]

        prompt = self._create_example_generation_prompt(num_examples)

        try:
            response = completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in Indic language transliteration.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            # Parse the response
            examples = self._parse_examples_response(_response_text(response))

            # Cache the examples
            if self.cache_examples:
                self._examples_cache[cache_key] = examples

            return examples[:num_examples]

        except Exception as e:
            logger.warning("Could not generate few-shot examples: %s", e)
            return self._get_fallback_examples()

    def _create_example_generation_prompt(self, num_examples: int) -> str:
        """Create prompt for generating few-shot examples."""
        source_native = self.INDIC_LANGUAGES[self.source_lang]["native"]
        target_native = self.INDIC_LANGUAGES[self.target_lang]["native"]

        if self.source_lang == "hindi" and self.target_lang == "english":
            return f"""Generate {num_examples} transliteration examples from Hindi to English.
Include diverse examples: proper names, cities, cultural terms, common words.

Format each example as:
source: [Hindi text]
target: [English transliteration]

Examples should include:
- Common Indian names (e.g., राजेश → Rajesh)
- Major cities (e.g., मुंबई → Mumbai)
- Cultural terms (e.g., दीवाली → Diwali)
- Common greetings (e.g., नमस्ते → Namaste)
- States or regions (e.g., राजस्थान → Rajasthan)"""

        if self.source_lang == "tamil" and self.target_lang == "english":
            return f"""Generate {num_examples} transliteration examples from Tamil to English.
Include diverse examples: Tamil names, cities, cultural terms.

Format each example as:
source: [Tamil text]
target: [English transliteration]

Examples should include:
- Tamil names (e.g., முருகன் → Murugan)
- Cities (e.g., சென்னை → Chennai)
- Cultural terms (e.g., பொங்கல் → Pongal)
- Common words"""

        # Generic prompt for other language pairs
        return f"""Generate {num_examples} transliteration examples from {source_native} ({self.source_lang}) to {target_native} ({self.target_lang}).
This is TRANSLITERATION (phonetic conversion), not translation.

Format each example as:
source: [source text]
target: [transliterated text]

Include diverse examples: names, places, cultural terms, common words.
Focus on accurate phonetic representation."""

    def _parse_examples_response(self, response: str) -> list[dict[str, str]]:
        """Parse LLM response to extract examples."""
        examples = []
        lines = response.strip().split("\n")

        current_example: dict[str, str] = {}
        for line in lines:
            line = line.strip()
            if line.startswith("source:"):
                if current_example:
                    examples.append(current_example)
                    current_example = {}
                current_example["source"] = line.replace("source:", "").strip()
            elif line.startswith("target:"):
                current_example["target"] = line.replace("target:", "").strip()

        if (
            current_example
            and "source" in current_example
            and "target" in current_example
        ):
            examples.append(current_example)

        return examples

    def _get_fallback_examples(self) -> list[dict[str, str]]:
        """Get fallback examples if generation fails."""
        if self.source_lang == "hindi" and self.target_lang == "english":
            return [
                {"source": "राज", "target": "Raj"},
                {"source": "भारत", "target": "Bharat"},
                {"source": "नमस्ते", "target": "Namaste"},
                {"source": "दिल्ली", "target": "Delhi"},
                {"source": "गंगा", "target": "Ganga"},
            ]
        if self.source_lang == "tamil" and self.target_lang == "english":
            return [
                {"source": "தமிழ்", "target": "Tamil"},
                {"source": "சென்னை", "target": "Chennai"},
                {"source": "வணக்கம்", "target": "Vanakkam"},
            ]
        return []

    def transliterate(
        self,
        text: str,
        use_few_shot: bool = True,
        num_examples: int = 5,
    ) -> str:
        """Transliterate text from source language to target language.

        Args:
            text: Text to transliterate.
            use_few_shot: Whether to use few-shot examples.
            num_examples: Number of few-shot examples to use.

        Returns:
            Transliterated text.

        Raises:
            RuntimeError: If the LLM transliteration call fails.
        """
        if not text or not text.strip():
            return ""

        # Get few-shot examples if requested
        examples = []
        if use_few_shot:
            examples = self.generate_few_shot_examples(num_examples)

        # Create the transliteration prompt
        system_prompt = self._create_system_prompt(examples)
        user_prompt = f"Transliterate the following text:\n{text}"

        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=len(text) * 3,  # Rough estimate for transliteration length
            )

            result: str = _response_text(response).strip()

            # Clean up the result (remove any explanation if present)
            if "\n" in result:
                # Take the first line if there are multiple lines
                result = result.split("\n")[0].strip()

            return result

        except Exception as e:
            logger.error("Transliteration failed: %s", e)
            raise RuntimeError(f"Failed to transliterate text: {e}") from e

    def _create_system_prompt(self, examples: list[dict[str, str]]) -> str:
        """Create the system prompt for transliteration."""
        source_native = self.INDIC_LANGUAGES[self.source_lang]["native"]
        target_native = self.INDIC_LANGUAGES[self.target_lang]["native"]

        prompt = f"""You are an expert in Indic language transliteration.
Task: Convert {source_native} ({self.source_lang}) text to {target_native} ({self.target_lang}) phonetic representation.

Important Rules:
1. This is TRANSLITERATION (sound/phonetic conversion), NOT translation
2. Preserve the pronunciation of the original text
3. Use commonly accepted romanization for Indian languages
4. For proper nouns (names, places), use standard spellings when known
5. Handle special characters correctly (anusvara ं, visarga ः, nukta ़, etc.)
6. Output ONLY the transliterated text, no explanations
7. Use plain ASCII English letters only (a-z) and spaces. Do NOT use diacritics, accents, or macrons (write "patubala", never "pātubālā"; "anjana", never "añjana")"""

        if examples:
            prompt += "\n\nExamples:"
            for ex in examples:
                prompt += f"\n{ex['source']} → {ex['target']}"

        return prompt

    def default_max_tokens_for(self, texts: list[str]) -> int:
        """Estimate max output tokens for transliterating ``texts`` as a group."""
        return max(64, sum(len(t) for t in texts) * 3 + 8 * len(texts))

    def build_group_messages(
        self, texts: list[str], examples: list[dict[str, str]] | None = None
    ) -> list[dict[str, str]]:
        """Build chat messages for transliterating a numbered group of texts.

        Shared by the synchronous ``transliterate_batch`` and the async Batch-API
        path (``indicate.batch``) so both produce identical prompts.
        """
        batch_text = "\n".join(f"{j + 1}. {text}" for j, text in enumerate(texts))
        system_prompt = self._create_system_prompt(examples or [])
        user_prompt = (
            "Transliterate each of the following texts "
            "(output one transliteration per line, numbered):\n"
            f"{batch_text}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def transliterate_batch(
        self,
        texts: list[str],
        batch_size: int = 10,
        use_few_shot: bool = True,
    ) -> list[str]:
        """Transliterate multiple texts efficiently.

        Args:
            texts: List of texts to transliterate.
            batch_size: Number of texts to process in one API call.
            use_few_shot: Whether to use few-shot examples.

        Returns:
            List of transliterated texts.
        """
        results = []

        # Get few-shot examples once
        examples = []
        if use_few_shot:
            examples = self.generate_few_shot_examples()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            messages = self.build_group_messages(batch, examples)

            try:
                response = completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )

                # Parse batch response
                batch_results = self._parse_batch_response(
                    _response_text(response), len(batch)
                )
                results.extend(batch_results)

            except Exception as e:
                logger.error("Batch transliteration failed: %s", e)
                # Fallback to individual transliteration
                for text in batch:
                    try:
                        result = self.transliterate(text, use_few_shot=False)
                        results.append(result)
                    except Exception:
                        results.append("")  # Empty string for failures

        return results

    def _parse_batch_response(self, response: str, expected_count: int) -> list[str]:
        """Parse batch transliteration response."""
        results = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            # Remove numbering if present (e.g., "1. ", "2. ")
            if line and line[0].isdigit() and ". " in line:
                line = line.split(". ", 1)[1] if ". " in line else line

            if line:
                results.append(line)

        # Ensure we have the right number of results
        while len(results) < expected_count:
            results.append("")

        return results[:expected_count]
