from __future__ import annotations

import os
from importlib.resources import files
from typing import TYPE_CHECKING, Self, cast

from safetensors.torch import load_file

from .decoder import Decoder
from .encoder import Encoder
from .logging import get_logger
from .utils import CharTokenizer, batch_candidates, load_tokenizer

if TYPE_CHECKING:
    from .rerank import Reranker

logger = get_logger()


def _nbest_combine(word_cands: list[list[tuple[str, float]]], n: int) -> list[str]:
    """Beam-combine per-word candidate lists into up to ``n`` ranked phrases."""
    combos: list[tuple[str, float]] = [("", 0.0)]
    for c in word_cands:
        cands = c[:n] or [("", 0.0)]
        combos = [
            (p + (" " if p else "") + t, s + sc) for p, s in combos for t, sc in cands
        ]
        combos.sort(key=lambda x: x[1], reverse=True)
        combos = combos[:n]
    return [p for p, _ in combos][:n]


class Seq2SeqTransliterator:
    """Base char-level seq2seq transliterator: lazy-loaded singleton.

    Subclasses set ``SUBDIR`` (the per-language folder) and the tokenizer
    filenames. Files are resolved local-first (``indicate/data/<SUBDIR>/...`` —
    present after training) and otherwise **downloaded from the HF model repo**
    (``HF_REPO`` @ ``HF_REVISION``) and cached, so the wheel ships no weights.
    All mutable state is assigned on the concrete subclass, so each language's
    model loads and caches independently.
    """

    # HF model repo holding per-language dirs (tokenizers + saved_weights/).
    HF_REPO: str = "soodoku/indicate"
    HF_REVISION: str = "v0.7.0"

    SUBDIR: str = ""  # e.g. "hindi_to_english"
    INPUT_VOCAB: str = ""  # tokenizer filename, e.g. "hindi_tokens.json"
    TARGET_VOCAB: str = ""  # "english_tokens.json"
    ENCODER_FILE: str = "saved_weights/encoder.safetensors"
    DECODER_FILE: str = "saved_weights/decoder.safetensors"

    embedding_dim: int = 256
    units: int = 1024

    max_length_input: int = 64
    max_length_output: int = 64
    START_TOKEN: str = "^"  # noqa: S105 - seq2seq framing token, not a secret
    END_TOKEN: str = "$"  # noqa: S105 - seq2seq framing token, not a secret

    # Decoding: 1 = greedy; >1 enables length-normalized beam search. Beam search
    # is the shipped default (+1.5 pts exact-match on Hindi Dakshina vs greedy).
    BEAM_WIDTH: int = 5
    # Optional LM re-ranker for beam candidates (opt-in; None = ship default).
    RERANKER: Reranker | None = None
    # True only for weights trained with attention padding masks (train.py masks
    # by default); must match the shipped weights or inference degrades.
    MASK_PADDING: bool = False

    _instance: Seq2SeqTransliterator | None = None
    _weights_loaded: bool = False

    input_lang_tokenizer: CharTokenizer | None = None
    target_lang_tokenizer: CharTokenizer | None = None
    encoder: Encoder | None = None
    decoder: Decoder | None = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cast(Self, cls._instance)

    @classmethod
    def _local(cls, rel: str) -> str:
        """Local path for a file under ``indicate/data/<SUBDIR>/`` (may not exist)."""
        return os.path.join(str(files(__package__)), "data", cls.SUBDIR, rel)

    @classmethod
    def _resolve(cls, rel: str) -> str:
        """Resolve a model file: local if present, else download from HF + cache."""
        local = cls._local(rel)
        if os.path.exists(local):
            return local
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            cls.HF_REPO, f"{cls.SUBDIR}/{rel}", revision=cls.HF_REVISION
        )

    @classmethod
    def get_model_path(cls) -> str:
        """Local saved_weights dir (for display; weights may be on HF instead)."""
        return os.path.dirname(cls._local(cls.ENCODER_FILE))

    @classmethod
    def get_input_vocab(cls) -> str:
        return cls._resolve(cls.INPUT_VOCAB)

    @classmethod
    def get_target_vocab(cls) -> str:
        return cls._resolve(cls.TARGET_VOCAB)

    @classmethod
    def _load_weights(cls) -> None:
        try:
            cls.input_lang_tokenizer = load_tokenizer(cls._resolve(cls.INPUT_VOCAB))
            cls.target_lang_tokenizer = load_tokenizer(cls._resolve(cls.TARGET_VOCAB))

            cls.encoder = Encoder(
                cls.input_lang_tokenizer.vocab_size, cls.embedding_dim, cls.units
            )
            cls.decoder = Decoder(
                cls.target_lang_tokenizer.vocab_size, cls.embedding_dim, cls.units
            )

            cls.encoder.load_state_dict(load_file(cls._resolve(cls.ENCODER_FILE)))
            cls.decoder.load_state_dict(load_file(cls._resolve(cls.DECODER_FILE)))
            cls.encoder.eval()
            cls.decoder.eval()
            cls._weights_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    @classmethod
    def transliterate(cls, input: str, n: int = 1) -> str | list[str]:
        """Transliterate one text (thin wrapper over ``transliterate_batch``).

        Args:
            input: source-language text
            n: number of candidates. ``n == 1`` (default) returns a single
                best string; ``n > 1`` returns a list of up to ``n`` ranked
                candidates (requires beam search).

        Returns:
            str when ``n == 1``; list[str] when ``n > 1``.

        Raises:
            TypeError: If input is None
            ValueError: If input is not a string
        """
        if input is None:
            raise TypeError("Input cannot be None")
        if not isinstance(input, str):
            raise ValueError("Input must be a string")
        return cls.transliterate_batch([input], n)[0]

    @classmethod
    def transliterate_batch(
        cls, inputs: list[str], n: int = 1
    ) -> list[str] | list[list[str]]:
        """Transliterate many texts at once — the batched decode engine.

        All words across all inputs are decoded in a single batch (one encoder /
        decoder pass per step), which is much faster than calling ``transliterate``
        per item. Returns one result per input, aligned to ``inputs``: a ``str``
        each when ``n == 1``, else a ``list[str]`` of up to ``n`` candidates each.
        """
        if not cls._weights_loaded:
            cls._load_weights()
        assert cls.input_lang_tokenizer is not None  # noqa: S101 - narrowing
        assert cls.target_lang_tokenizer is not None  # noqa: S101 - narrowing
        assert cls.encoder is not None  # noqa: S101 - narrowing
        assert cls.decoder is not None  # noqa: S101 - narrowing

        # Split each text into words (decoded independently to avoid drift); keep
        # the per-text spans so we can reassemble.
        per_text_words: list[list[str]] = []
        flat: list[str] = []
        for text in inputs:
            words = text.split(" ") if isinstance(text, str) and text.strip() else []
            per_text_words.append(words)
            flat.extend(words)

        beam = max(cls.BEAM_WIDTH, n) if n > 1 else cls.BEAM_WIDTH
        try:
            cand_lists = (
                batch_candidates(
                    flat,
                    cls.input_lang_tokenizer,
                    cls.target_lang_tokenizer,
                    cls.encoder,
                    cls.decoder,
                    cls.max_length_input,
                    cls.max_length_output,
                    beam,
                    cls.MASK_PADDING,
                )
                if flat
                else []
            )
        except Exception as exe:
            logger.error(f"Batch transliteration failed: {exe}")
            cand_lists = [[] for _ in flat]

        results: list = []
        i = 0
        for words in per_text_words:
            wc = cand_lists[i : i + len(words)]
            i += len(words)
            if n <= 1:
                parts = []
                for c in wc:
                    if not c:
                        parts.append("")
                    elif cls.RERANKER is not None and beam > 1:
                        parts.append(cls.RERANKER.best(c))
                    else:
                        parts.append(c[0][0])
                results.append(" ".join(parts))
            else:
                results.append([] if not words else _nbest_combine(wc, n))
        return results
