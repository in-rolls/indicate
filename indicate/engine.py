"""Backends, and the order they are tried in.

A word is resolved by the first backend that will answer it. ``lookup`` reads a
table, ``model`` decodes with the local seq2seq weights, ``llm`` asks a provider.
The chain is ordinary data::

    ("lookup", "model")           # the default: table, then decode the tail
    ("model",)                    # what a benchmark must use
    ("lookup",)                   # table only -- is my corpus already covered?
    ("lookup", "llm")             # the table intercepts the paid path
    ("lookup", "model", "llm")    # decode locally, escalate only what is left

This generalizes the hit/miss/reassemble loop the transliterator already ran; it
is not a new concept, just one that can now be spelled.

**Declining is the whole protocol.** :meth:`Backend.resolve` returns one entry
per word: a candidate list, or ``None``/``[]`` meaning *I will not answer this*.
Treating an empty list as a decline is a real change -- a decoder exception used
to yield ``[]`` and become an empty string in the output, silently. Now it falls
through, so ``("lookup", "llm", "model")`` degrades instead of losing a word.

**Declining is not the same as being unavailable.** A backend that loaded its
table and had no entry for a word has *declined* -- ordinary, silent, and the
whole point of a chain. A backend that could not obtain its asset at all is
*unavailable*, and if every backend in the chain is unavailable the caller is
about to receive an empty string that looks like an answer. That case raises
:class:`BackendsUnavailableError` instead. The distinction matters because
``engine=["lookup"]`` over an uncovered corpus declines every word by design,
and must stay quiet.

**Nothing heavy is imported until a backend actually runs.** torch lives inside
:meth:`ModelBackend.resolve` and litellm inside :meth:`LLMBackend.resolve`, so a
text whose every word hits the table imports neither. That property is worth
about 4x on cold start and is asserted in ``tests/test_lookup_bench.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .languages import BACKENDS, Pair, UnsupportedPairError, supports
from .logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .rerank import Reranker

logger = get_logger()

#: One romanization and its score. Scores are comparable only within a backend.
Candidates = list[tuple[str, float]]

#: The chain used when a caller does not choose one.
DEFAULT_ENGINE: tuple[str, ...] = ("lookup", "model")

#: Valid backend names, re-exported so callers validate against one set.
KNOWN = BACKENDS

#: Score attached to an answer from a backend that does not rank.
#:
#: A table or an LLM returns exactly one candidate, so this value never orders
#: anything *within* a word. It exists so phrase assembly works: beam scores are
#: length-normalized log-probs and therefore negative, and ``0.0`` sits above
#: them, which keeps a single-candidate word from dragging a phrase down. It is
#: not a probability and must never be compared across backends.
AUTHORITATIVE = 0.0


class BackendsUnavailableError(RuntimeError):
    """No backend in the chain could obtain what it needed to answer anything.

    Raised rather than returning ``""``, which is indistinguishable from a
    legitimate empty answer and exits zero.
    """


@runtime_checkable
class Backend(Protocol):
    """Something that can answer some words and decline the rest."""

    name: str
    #: Set by ``resolve`` when the backend could not obtain its asset at all.
    unavailable: bool

    def resolve(self, words: Sequence[str]) -> Sequence[Candidates | None]:
        """Answer what it can.

        Args:
            words: Words still unresolved, in order.

        Returns:
            One entry per word: candidates best-first, or ``None`` to decline.
        """
        ...


class LookupBackend:
    """Answers from the packaged word table. No torch, no network."""

    name = "lookup"

    def __init__(self, pair: Pair) -> None:
        """Bind to one language pair.

        Args:
            pair: The direction to load a table for.
        """
        self._pair = pair
        self._table = None
        self._loaded = False
        self.unavailable = False

    @property
    def table(self):
        """The loaded table, or ``None`` when this pair ships none."""
        if not self._loaded:
            from .lookup import Lookup

            self._table = Lookup.load(self._pair.subdir)
            self._loaded = True
        return self._table

    def resolve(self, words: Sequence[str]) -> list[Candidates | None]:
        """Return the table's answer for each word it knows.

        Args:
            words: Words to look up.

        Returns:
            A single authoritative candidate per hit, ``None`` per miss.
        """
        table = self.table
        if table is None:
            self.unavailable = True
            return [None] * len(words)
        out: list[Candidates | None] = []
        for word in words:
            hit = table.get(word)
            out.append([(hit, AUTHORITATIVE)] if hit is not None else None)
        return out


class ModelBackend:
    """Decodes with the local seq2seq weights."""

    name = "model"

    def __init__(
        self,
        pair: Pair,
        *,
        beam: int = 5,
        reranker: Reranker | None = None,
    ) -> None:
        """Bind to one language pair.

        Args:
            pair: The direction to load weights for.
            beam: Beam width; 1 is greedy.
            reranker: Optional LM re-ranker. Applied here rather than during
                assembly so it structurally can only ever see model candidates.
        """
        self._pair = pair
        self._beam = beam
        self._reranker = reranker
        self.unavailable = False

    def resolve(self, words: Sequence[str]) -> list[Candidates | None]:
        """Decode every word in one batch.

        Args:
            words: Words to decode.

        Returns:
            Candidates per word; ``None`` for any word the decoder dropped, so
            it can fall through instead of becoming an empty string.
        """
        if not words:
            return []
        # Imported here, not at module scope: this is the line that keeps torch
        # out of a process whose words all hit the table.
        from .transliterator import model_for

        try:
            # Obtaining the model is inside the try as well as decoding with it:
            # a machine with no weights and no network fails here, and that is a
            # decline, not a crash.
            decoded = model_for(self._pair).candidates(list(words), self._beam)
        except Exception as exc:
            logger.error(f"decode failed for {len(words)} word(s): {exc}")
            self.unavailable = True
            return [None] * len(words)

        out: list[Candidates | None] = []
        for candidates in decoded:
            if not candidates:
                out.append(None)
            elif self._reranker is not None and self._beam > 1:
                best = self._reranker.best(candidates)
                out.append(
                    [c for c in candidates if c[0] == best]
                    + [c for c in candidates if c[0] != best]
                )
            else:
                out.append(list(candidates))
        return out


class LLMBackend:
    """Asks a provider. Deduplicates first, because this one costs money."""

    name = "llm"

    def __init__(
        self,
        pair: Pair,
        *,
        transliterator=None,
        group_size: int = 25,
        **client_kwargs,
    ) -> None:
        """Bind to one language pair.

        Args:
            pair: The direction to transliterate.
            transliterator: An existing ``IndicLLMTransliterator`` to reuse.
            group_size: Words per request.
            **client_kwargs: Passed to ``IndicLLMTransliterator`` (``provider``,
                ``model``, ``api_key``, ``temperature``).
        """
        self._pair = pair
        self._client = transliterator
        self._group_size = group_size
        self._client_kwargs = client_kwargs
        self.unavailable = False

    def client(self):
        """The provider client, constructed on first use."""
        if self._client is None:
            # litellm costs ~1.3s to import; never pay it for a table hit.
            from .llm_indic import IndicLLMTransliterator

            self._client = IndicLLMTransliterator(
                self._pair.source, self._pair.target, **self._client_kwargs
            )
        return self._client

    def resolve(self, words: Sequence[str]) -> list[Candidates | None]:
        """Transliterate the distinct words in as few requests as possible.

        A failure declines the whole group rather than raising, so a chain like
        ``("lookup", "llm", "model")`` falls back to local decoding when the
        network or the API key is not there.

        Args:
            words: Words to transliterate.

        Returns:
            One authoritative candidate per word the provider answered.
        """
        if not words:
            return []
        unique = list(dict.fromkeys(words))
        try:
            answers = self.client().transliterate_batch(
                unique, batch_size=self._group_size
            )
        except Exception as exc:
            logger.error(f"LLM backend declined {len(unique)} word(s): {exc}")
            self.unavailable = True
            return [None] * len(words)
        answered = {
            word: answer.strip()
            for word, answer in zip(unique, answers, strict=False)
            if answer and answer.strip()
        }
        return [
            [(answered[word], AUTHORITATIVE)] if word in answered else None
            for word in words
        ]


def normalize_engine(engine: Sequence[str] | str | None) -> tuple[str, ...]:
    """Coerce an engine argument to a validated tuple of backend names.

    Args:
        engine: A backend name, a sequence of them, or ``None`` for the default.

    Returns:
        The chain, in order.

    Raises:
        ValueError: If a name is not a known backend, or the chain is empty.
    """
    if engine is None:
        return DEFAULT_ENGINE
    names = (engine,) if isinstance(engine, str) else tuple(engine)
    if not names:
        raise ValueError("engine must name at least one backend")
    unknown = [n for n in names if n not in KNOWN]
    if unknown:
        raise ValueError(
            f"unknown backend(s) {unknown}; known: {', '.join(sorted(KNOWN))}"
        )
    return names


def build(
    engine: Sequence[str] | str | None,
    pair: Pair,
    *,
    beam: int = 5,
    reranker: Reranker | None = None,
    llm=None,
    **llm_kwargs,
) -> list[Backend]:
    """Construct the backends for a chain, without running any of them.

    Backends this pair has no support for are dropped rather than failing, so
    ``("lookup", "model")`` still works for a language that ships no table. If
    that leaves nothing, the direction is genuinely unsupported and saying so is
    better than falling through to a backend the caller did not ask to pay for.

    Args:
        engine: Backend names in order, or ``None`` for the default.
        pair: The direction being transliterated.
        beam: Beam width for the model backend.
        reranker: Optional re-ranker for the model backend.
        llm: An existing LLM client to reuse.
        **llm_kwargs: Provider settings for the LLM backend.

    Returns:
        Constructed backends, in chain order.

    Raises:
        UnsupportedPairError: If no backend in the chain supports this direction.
    """
    names = normalize_engine(engine)
    backends: list[Backend] = []
    for name in names:
        if not supports(pair.source, pair.target, name):
            continue
        if name == "lookup":
            backends.append(LookupBackend(pair))
        elif name == "model":
            backends.append(ModelBackend(pair, beam=beam, reranker=reranker))
        else:
            backends.append(LLMBackend(pair, transliterator=llm, **llm_kwargs))
    if not backends:
        raise UnsupportedPairError(
            f"no backend in {list(names)} supports "
            f"{pair.source}->{pair.target}; "
            f"try engine=['llm'] or see indicate.supported()"
        )
    return backends


def resolve_words(
    words: Sequence[str], backends: Sequence[Backend]
) -> list[Candidates]:
    """Run the chain: each backend sees only what the previous ones declined.

    Args:
        words: Words to resolve, in order.
        backends: The chain.

    Returns:
        One candidate list per word, aligned to ``words``. A word every backend
        declined gets ``[]``.

    Raises:
        BackendsUnavailableError: If there were words, nothing was resolved, and
            every backend reported itself unavailable -- i.e. the caller is
            about to get an empty string not because the answer is empty but
            because nothing was able to run.
    """
    out: list[Candidates] = [[] for _ in words]
    pending = list(range(len(words)))
    for backend in backends:
        if not pending:
            break
        answers = backend.resolve([words[i] for i in pending])
        still: list[int] = []
        for index, answer in zip(pending, answers, strict=True):
            if answer:
                out[index] = list(answer)
            else:
                still.append(index)
        pending = still

    if words and len(pending) == len(words) and _all_unavailable(backends):
        raise BackendsUnavailableError(
            "nothing could answer "
            f"{len(words)} word(s): "
            + "; ".join(f"{b.name} {_why(b)}" for b in backends)
        )
    return out


def _all_unavailable(backends: Sequence[Backend]) -> bool:
    """Report whether every backend failed to obtain what it needed."""
    return bool(backends) and all(
        getattr(backend, "unavailable", False) for backend in backends
    )


def _why(backend: Backend) -> str:
    """Return an actionable phrase for why a backend could not run."""
    if backend.name == "lookup":
        return (
            "has no table (build one with "
            "training/build_lookup.py, or omit it from the engine)"
        )
    if backend.name == "model":
        return "could not load weights (check the network or the HF cache)"
    return "could not reach the provider (check the API key and the network)"
