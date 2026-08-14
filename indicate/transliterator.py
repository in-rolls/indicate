"""The local seq2seq decoder, as one loadable object per language pair.

This used to be a class hierarchy -- ``Seq2SeqTransliterator`` with a subclass
per direction, holding its weights in *class* attributes on a ``__new__``-cached
singleton. Adding a language meant adding a module, and the mutable class state
leaked between callers badly enough that the tests had to restore it in
``tearDown``.

Now a direction is a :class:`~indicate.languages.Pair` -- data -- and the loaded
weights are an instance, cached per pair in :data:`_MODELS`. Adding a language is
a dict entry.

This module deliberately knows nothing about lookup tables, LLMs or chains. It
decodes words with torch and stops there; :mod:`indicate.engine` composes it with
everything else. torch is imported inside :meth:`Seq2SeqModel.load`, so importing
this module costs nothing.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .languages import Pair
from .logging import get_logger
from .resources import HF_REPO, HF_REVISION, local_data_path, resolve_data

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .decoder import Decoder
    from .encoder import Encoder
    from .utils import CharTokenizer

logger = get_logger()

#: Architecture of the shipped weights. Not per-pair: both languages were
#: trained with the same encoder/decoder shape, and changing either would need
#: new weights rather than a new setting.
EMBEDDING_DIM = 256
UNITS = 1024

#: Beam width when a caller does not pick one. Beam search is the shipped
#: default: +1.5 points exact-match on Hindi Dakshina against greedy.
DEFAULT_BEAM = 5

ENCODER_FILE = "saved_weights/encoder.safetensors"
DECODER_FILE = "saved_weights/decoder.safetensors"


class Seq2SeqModel:
    """Char-level encoder/decoder weights for one language pair, loaded lazily.

    Files resolve local-first (``indicate/data/<subdir>/``, present after
    training) and otherwise download from the Hugging Face model repo and cache,
    so the wheel ships no weights.
    """

    def __init__(self, pair: Pair, *, mask_padding: bool = False) -> None:
        """Bind to a pair without loading anything.

        Args:
            pair: The direction these weights transliterate.
            mask_padding: True only for weights trained with attention padding
                masks. Must match the weights or inference degrades.
        """
        self.pair = pair
        self.mask_padding = mask_padding
        self.input_tokenizer: CharTokenizer | None = None
        self.target_tokenizer: CharTokenizer | None = None
        self.encoder: Encoder | None = None
        self.decoder: Decoder | None = None
        self._loaded = False

    def path(self, rel: str) -> str:
        """Resolve one of this pair's files, downloading it if necessary.

        Args:
            rel: Path relative to the pair's data directory.

        Returns:
            A path on disk.
        """
        return resolve_data(self.pair.subdir, rel, repo=HF_REPO, revision=HF_REVISION)

    @property
    def weights_dir(self) -> str:
        """Local weights directory, for display. Weights may live on HF instead."""
        return os.path.dirname(local_data_path(self.pair.subdir, ENCODER_FILE))

    def load(self) -> None:
        """Load tokenizers and weights. Idempotent.

        Raises:
            RuntimeError: If any file is missing or fails to load.
        """
        if self._loaded:
            return
        # torch enters the process here and nowhere earlier.
        from safetensors.torch import load_file

        from .decoder import Decoder
        from .encoder import Encoder
        from .utils import load_tokenizer

        try:
            self.input_tokenizer = load_tokenizer(self.path(self.pair.input_vocab))
            self.target_tokenizer = load_tokenizer(self.path(self.pair.target_vocab))
            self.encoder = Encoder(
                self.input_tokenizer.vocab_size, EMBEDDING_DIM, UNITS
            )
            self.decoder = Decoder(
                self.target_tokenizer.vocab_size, EMBEDDING_DIM, UNITS
            )
            self.encoder.load_state_dict(load_file(self.path(ENCODER_FILE)))
            self.decoder.load_state_dict(load_file(self.path(DECODER_FILE)))
            self.encoder.eval()
            self.decoder.eval()
            self._loaded = True
        except Exception as exc:
            raise RuntimeError(f"Failed to load model: {exc}") from exc

    def candidates(
        self, words: Sequence[str], beam: int = DEFAULT_BEAM
    ) -> list[list[tuple[str, float]]]:
        """Decode words in one batch.

        Args:
            words: Source-language words.
            beam: Beam width; 1 is greedy.

        Returns:
            Scored candidates per word, best first.
        """
        from .utils import batch_candidates

        self.load()
        assert self.input_tokenizer is not None  # noqa: S101 - narrowing
        assert self.target_tokenizer is not None  # noqa: S101 - narrowing
        assert self.encoder is not None  # noqa: S101 - narrowing
        assert self.decoder is not None  # noqa: S101 - narrowing
        return batch_candidates(
            list(words),
            self.input_tokenizer,
            self.target_tokenizer,
            self.encoder,
            self.decoder,
            self.pair.max_input,
            self.pair.max_output,
            beam,
            self.mask_padding,
        )


#: One loaded model per pair, so weights are read once per process.
_MODELS: dict[tuple[str, str], Seq2SeqModel] = {}


def model_for(pair: Pair) -> Seq2SeqModel:
    """Return the cached model for a pair, creating it if needed.

    Creating it does not load weights; that happens on first decode.

    Args:
        pair: The direction wanted.

    Returns:
        The shared model object for that pair.
    """
    model = _MODELS.get(pair.key)
    if model is None:
        model = Seq2SeqModel(pair)
        _MODELS[pair.key] = model
    return model


def clear_models() -> None:
    """Drop every loaded model. Intended for tests."""
    _MODELS.clear()
