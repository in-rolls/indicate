"""Batch-mode LLM transliteration via provider Batch APIs (batchlane).

The synchronous :class:`~indicate.llm_indic.IndicLLMTransliterator` issues one
``litellm.completion`` call per request. For transliterating the millions of unique
tokens in an electoral roll, that is slow and expensive. This module routes the work
through a provider's asynchronous **Batch API** instead (~50% cheaper), with
checkpointing and resume.

Because a batch can take up to 24h to finish, *submit* and *collect* are separate
steps; :func:`transliterate_tokens_batched` is a convenience driver that submits then
polls to completion. State is durable, so a killed process resumes from the checkpoint
rather than resubmitting finished work:

* ``checkpoint_path`` -- a JSONL file of resolved ``{"token", "translit"}`` pairs
  (append-only; the durable result map). Kept dependency-free (no pandas/pyarrow).
* ``checkpoint_path + ".batchstate.json"`` -- in-flight batch ids and the
  ``custom_id -> [tokens]`` mapping needed to align results back to tokens.

Provider submission, request limits, polling, and response normalization use
``batchlane``. Transliteration prompts, local fallback, result validation, and
per-token retries remain here. The submission journal at
``checkpoint_path + ".batchlane.jsonl"`` records provider handles; the state file
also saves the exact prompts before submission so a restart does not regenerate
them. Use one process per checkpoint.

**The cheapest request is the one you do not send.** Before submitting anything,
:func:`submit_transliteration_batches` runs the local part of ``engine`` --
everything before ``"llm"`` -- and writes what it answers straight to the
checkpoint. Measured on 1M rows of the Punjab roll with the default
``("lookup", "llm")``:

=========================  ==============  ===========
                           llm only        with lookup
=========================  ==============  ===========
unique tokens to resolve   46,902          1,497
batch requests submitted   1,877           60
=========================  ==============  ===========

That is 96.8% of *unique* tokens, which is the number that matters here -- a
batch API deduplicates, so token frequency buys nothing. It is this high because
the roll shares a vocabulary with the corpus the table was built from; general
text would see much less.

``engine=("lookup", "model", "llm")`` goes further and decodes the table's
misses locally, submitting only what both decline. ``engine=("llm",)`` submits
everything.
"""

from __future__ import annotations

import itertools
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import batchlane as bl

from .llm_indic import IndicLLMTransliterator
from .logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger()

DEFAULT_GROUP_SIZE = 25
DEFAULT_MAX_REQUESTS_PER_BATCH = 50_000

#: Answer from the table first, submit only what it declines.
DEFAULT_BATCH_ENGINE: tuple[str, ...] = ("lookup", "llm")


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
@dataclass
class BatchJob:
    """One submitted batch (a provider batch id + the tokens it covers)."""

    handle: str
    custom_id_to_tokens: dict[str, list[str]]
    status: str = "submitted"  # "submitted" | "done"

    @property
    def batch_id(self) -> str:
        return bl.BatchHandle.from_json(self.handle).job_id


@dataclass
class BatchState:
    """Durable record of an in-flight transliteration run."""

    provider: str
    model: str
    source_lang: str
    target_lang: str
    group_size: int
    temperature: float
    use_few_shot: bool
    jobs: list[BatchJob] = field(default_factory=list)
    submitted_at: float | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)
    completion_window: str | None = None
    max_requests_per_batch: int = DEFAULT_MAX_REQUESTS_PER_BATCH


def _state_path(checkpoint_path: Path) -> Path:
    return Path(str(checkpoint_path) + ".batchstate.json")


def _save_state(checkpoint_path: Path, state: BatchState) -> None:
    path = _state_path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(asdict(state), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_state(checkpoint_path: Path) -> BatchState | None:
    path = _state_path(checkpoint_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["jobs"] = [BatchJob(**job) for job in data["jobs"]]
    return BatchState(**data)


# --------------------------------------------------------------------------- #
# Resolved-pairs checkpoint (JSONL)
# --------------------------------------------------------------------------- #
def _load_resolved(checkpoint_path: Path) -> dict[str, str]:
    resolved: dict[str, str] = {}
    if not checkpoint_path.exists():
        return resolved
    with checkpoint_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            resolved[record["token"]] = record["translit"]
    return resolved


def _resolve_locally(
    tokens: list[str], source_lang: str, target_lang: str, engine: Sequence[str]
) -> dict[str, str]:
    """Answer whatever the local backends in ``engine`` can, before submitting.

    A batch API call costs money and returns in hours; a table read costs
    nothing and returns in nanoseconds. On roll-shaped input the table covers
    99% of tokens, so this is the difference between submitting a million
    requests and submitting ten thousand.

    Only the prefix of ``engine`` before ``"llm"`` runs here -- everything after
    it is what the provider is for. That is what makes
    ``("lookup", "model", "llm")`` mean "read the table, decode what is left
    locally, and submit only the residue".

    Args:
        tokens: Unique source tokens about to be submitted.
        source_lang: Normalized source language.
        target_lang: Normalized target language.
        engine: The full chain; the ``llm`` suffix is ignored.

    Returns:
        Token to romanization for the tokens answered locally.
    """
    from .engine import (
        BackendsUnavailableError,
        build,
        normalize_engine,
        resolve_words,
    )
    from .languages import (
        PAIRS,
        UnknownLanguageError,
        UnsupportedPairError,
    )
    from .languages import normalize as normalize_language

    # Validate the whole chain before anything else, and let it raise. A typo
    # like ("lookpu", "llm") used to be swallowed as "no local backend here",
    # after which every token was submitted to a paid provider. A misspelling
    # must cost an exception, not money.
    normalize_engine(engine)

    prefix = list(itertools.takewhile(lambda name: name != "llm", engine))
    if not prefix or not tokens:
        return {}

    # Normalized here, not at the call sites, so every caller gets it. PAIRS is
    # keyed on canonical names, and the llm path gets normalization for free
    # from IndicLLMTransliterator -- so without this the documented aliases
    # ("pa", "en") silently resolved nothing while the paid path handled them.
    try:
        source_lang = normalize_language(source_lang)
        target_lang = normalize_language(target_lang)
    except UnknownLanguageError:
        return {}

    pair = PAIRS.get((source_lang, target_lang))
    if pair is None:
        return {}
    try:
        backends = build(prefix, pair)
    except UnsupportedPairError as exc:
        logger.debug("no local backend for %s->%s: %s", source_lang, target_lang, exc)
        return {}
    try:
        resolved = resolve_words(tokens, backends)
    except BackendsUnavailableError as exc:
        # Nothing local could run -- no table built, no weights. That is fatal
        # for a bare `transliterate()` call, which has nothing else to try, but
        # here the `llm` suffix is exactly the fallback. Resolve nothing and let
        # every token be submitted. Without this, the default ("lookup", "llm")
        # chain aborted the whole batch on any machine with no lookup table,
        # which is every installed user.
        logger.debug("no local backend could run, submitting everything: %s", exc)
        return {}
    return {
        token: candidates[0][0]
        for token, candidates in zip(tokens, resolved, strict=True)
        if candidates
    }


def _append_resolved(checkpoint_path: Path, pairs: dict[str, str]) -> None:
    if not pairs:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as handle:
        handle.writelines(
            json.dumps({"token": token, "translit": translit}, ensure_ascii=False)
            + "\n"
            for token, translit in pairs.items()
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _chunk(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _make_transliterator(
    source_lang: str,
    target_lang: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    temperature: float,
) -> IndicLLMTransliterator:
    return IndicLLMTransliterator(
        source_lang,
        target_lang,
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
    )


def _journal_path(checkpoint_path: Path) -> Path:
    return Path(str(checkpoint_path) + ".batchlane.jsonl")


def _clear_submission(checkpoint_path: Path) -> None:
    # Remove the journal first: if interrupted, the completed state still
    # prevents another submission. Reversing this order could reuse old jobs.
    _journal_path(checkpoint_path).unlink(missing_ok=True)
    _state_path(checkpoint_path).unlink(missing_ok=True)


def _ensure_submitted(
    checkpoint_path: Path, state: BatchState, api_key: str | None = None
) -> None:
    if state.submitted_at is not None or not state.requests:
        return
    lines = [bl.BatchLine(**request) for request in state.requests]
    plan = bl.plan(lines, max_requests_per_batch=state.max_requests_per_batch)
    handles = bl.submit_all(
        lines,
        checkpoint=_journal_path(checkpoint_path),
        api_key=api_key,
        window=state.completion_window,
        max_requests_per_batch=state.max_requests_per_batch,
    )
    state.jobs = [
        BatchJob(
            handle=handle.to_json(),
            custom_id_to_tokens={
                line.custom_id: state.groups[line.custom_id] for line in chunk
            },
        )
        for handle, chunk in zip(handles, plan.chunks, strict=True)
    ]
    state.submitted_at = time.time()
    _save_state(checkpoint_path, state)


def _poll_job(job: BatchJob, api_key: str | None = None) -> tuple[str, dict[str, str]]:
    handle = bl.BatchHandle.from_json(job.handle)
    status = bl.status(handle, api_key=api_key)
    if not status.is_terminal:
        return "running", {}
    if status.state != "succeeded":
        return "failed", {}
    answers = {}
    seen = set()
    for result in bl.results(handle, api_key=api_key):
        if result.custom_id not in job.custom_id_to_tokens or result.custom_id in seen:
            raise ValueError(
                f"Unexpected or duplicate batch result ID: {result.custom_id!r}"
            )
        seen.add(result.custom_id)
        answer = bl.answer_text(result)
        if answer:
            answers[result.custom_id] = answer
    return "completed", answers


def _batch_provider(provider: str) -> str:
    return (
        "gemini" if provider in {"google", "google_ai_studio", "gemini"} else provider
    )


def _validate_resume(state: BatchState, transliterator: IndicLLMTransliterator) -> None:
    requested = (
        transliterator.source_lang,
        transliterator.target_lang,
        transliterator.model,
        transliterator.temperature,
        _batch_provider(transliterator.provider),
    )
    saved = (
        state.source_lang,
        state.target_lang,
        state.model,
        state.temperature,
        _batch_provider(state.provider),
    )
    if requested != saved:
        raise ValueError(
            "The batch checkpoint belongs to a different language pair, "
            "provider, model, or temperature."
        )


def submit_transliteration_batches(
    tokens: list[str],
    source_lang: str,
    target_lang: str,
    *,
    checkpoint_path: str | Path,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    completion_window: str | None = None,
    use_few_shot: bool = True,
    temperature: float = 0.3,
    max_requests_per_batch: int = DEFAULT_MAX_REQUESTS_PER_BATCH,
    engine: Sequence[str] = DEFAULT_BATCH_ENGINE,
) -> BatchState:
    """Submit unique ``tokens`` to the provider Batch API. Does not block.

    Already-resolved tokens (present in the checkpoint) and duplicates/blanks are
    skipped. Returns the persisted :class:`BatchState`.

    Backends in ``engine`` before ``"llm"`` run locally first; whatever they
    answer is written straight to the checkpoint and never submitted. Pass
    ``engine=("llm",)`` to send everything to the provider.
    """
    checkpoint_path = Path(checkpoint_path)
    if group_size < 1 or max_requests_per_batch < 1:
        raise ValueError("group_size and max_requests_per_batch must be positive")
    resolved = _load_resolved(checkpoint_path)

    seen: set[str] = set()
    todo: list[str] = []
    for token in tokens:
        if not token or token in seen or token in resolved:
            continue
        seen.add(token)
        todo.append(token)

    if "llm" not in engine:
        # A chain with no `llm` is a local-only probe -- engine=("lookup",) is
        # the documented "is my corpus already covered?" question. Submitting
        # its misses to a provider answers a question nobody asked and bills for
        # it, so resolve locally and stop. Ahead of _make_transliterator, which
        # would otherwise demand an API key for a run that needs none.
        local = _resolve_locally(todo, source_lang, target_lang, engine) if todo else {}
        if local:
            _append_resolved(checkpoint_path, local)
        logger.info(
            "engine=%s has no llm; answered %d of %d token(s) locally, "
            "submitted nothing",
            ",".join(engine),
            len(local),
            len(todo),
        )
        return BatchState(
            provider=provider or "none",
            model=model or "none",
            source_lang=source_lang,
            target_lang=target_lang,
            group_size=group_size,
            temperature=temperature,
            use_few_shot=use_few_shot,
            jobs=[],
        )

    transliterator = _make_transliterator(
        source_lang, target_lang, provider, model, api_key, temperature
    )
    provider = provider or transliterator.provider
    model = model or transliterator.model

    if todo:
        # After _make_transliterator so the language names are normalized.
        local = _resolve_locally(
            todo, transliterator.source_lang, transliterator.target_lang, engine
        )
        if local:
            _append_resolved(checkpoint_path, local)
            todo = [token for token in todo if token not in local]
            logger.info(
                "Local backends answered %d of %d token(s); submitting %d",
                len(local),
                len(local) + len(todo),
                len(todo),
            )

    state = _load_state(checkpoint_path)
    if state is not None:
        _validate_resume(state, transliterator)
        _ensure_submitted(checkpoint_path, state, api_key)
        return state

    state = BatchState(
        provider=provider,
        model=model,
        source_lang=transliterator.source_lang,
        target_lang=transliterator.target_lang,
        group_size=group_size,
        temperature=temperature,
        use_few_shot=use_few_shot,
        completion_window=completion_window,
        max_requests_per_batch=max_requests_per_batch,
    )
    if not todo:
        _save_state(checkpoint_path, state)
        return state

    lane = _batch_provider(provider)
    bl.get_adapter(lane)
    examples = transliterator.generate_few_shot_examples() if use_few_shot else []
    bare_model = model.removeprefix(f"{lane}/")
    batch_model = f"{lane}/{bare_model}"
    for index, group in enumerate(_chunk(todo, group_size)):
        custom_id = f"grp-{index}"
        params: dict[str, Any] = {"temperature": temperature}
        if lane == "gemini":
            params["thinking"] = {"type": "disabled", "budget_tokens": 0}
        else:
            params["max_completion_tokens"] = transliterator.default_max_tokens_for(
                group
            )
        request = bl.BatchLine(
            custom_id,
            batch_model,
            transliterator.build_group_messages(group, examples),
            params,
        )
        state.requests.append(asdict(request))
        state.groups[custom_id] = group

    _save_state(checkpoint_path, state)
    _ensure_submitted(checkpoint_path, state, api_key)
    return state


# --------------------------------------------------------------------------- #
# Collect
# --------------------------------------------------------------------------- #
def collect_transliteration_batches(
    checkpoint_path: str | Path,
    *,
    transliterator: IndicLLMTransliterator | None = None,
    api_key: str | None = None,
) -> tuple[bool, dict[str, str]]:
    """Poll in-flight batches once and append any completed results.

    Returns ``(all_done, resolved_map)``. A group whose result count does not match
    the request, or whose batch failed, is left out of ``resolved_map`` (the driver
    requeues such tokens). Safe to call repeatedly.
    """
    checkpoint_path = Path(checkpoint_path)
    state = _load_state(checkpoint_path)
    if state is None:
        return True, _load_resolved(checkpoint_path)

    if transliterator is None:
        transliterator = _make_transliterator(
            state.source_lang,
            state.target_lang,
            state.provider,
            state.model,
            api_key,
            state.temperature,
        )
    _validate_resume(state, transliterator)
    _ensure_submitted(checkpoint_path, state, api_key)

    all_done = True
    newly_resolved: dict[str, str] = {}
    for job in state.jobs:
        if job.status == "done":
            continue
        status, by_custom_id = _poll_job(job, api_key)

        if status == "completed":
            for custom_id, group in job.custom_id_to_tokens.items():
                text = by_custom_id.get(custom_id)
                if text is None:
                    continue  # missing -> requeued by the driver
                # Validate the raw line count *before* parsing: _parse_batch_response
                # pads/truncates to len(group), which would silently mis-align an
                # over- or under-count. Each token yields exactly one numbered line.
                raw_lines = [ln for ln in text.splitlines() if ln.strip()]
                # Deliberate reuse of the LLM transliterator's parser so batch
                # and interactive paths cannot drift on response handling.
                parsed = transliterator._parse_batch_response(  # noqa: SLF001
                    text, len(group)
                )
                if len(raw_lines) != len(group) or any(not p for p in parsed):
                    logger.warning(
                        "Batch %s group %s: %d output line(s) for %d tokens; "
                        "requeueing",
                        job.batch_id,
                        custom_id,
                        len(raw_lines),
                        len(group),
                    )
                    continue
                newly_resolved.update(zip(group, parsed, strict=True))
            job.status = "done"
        elif status == "failed":
            logger.warning(
                "Batch %s ended failed/cancelled/expired; tokens will be requeued",
                job.batch_id,
            )
            job.status = "done"
        else:
            all_done = False  # pending / running / finalizing

    _append_resolved(checkpoint_path, newly_resolved)
    _save_state(checkpoint_path, state)
    resolved = _load_resolved(checkpoint_path)
    return all_done, resolved


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def transliterate_tokens_batched(
    tokens: list[str],
    source_lang: str,
    target_lang: str,
    *,
    checkpoint_path: str | Path,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    completion_window: str | None = None,
    use_few_shot: bool = True,
    temperature: float = 0.3,
    poll_interval: float = 60.0,
    max_wait: float | None = None,
    requeue_passes: int = 2,
    max_requests_per_batch: int = DEFAULT_MAX_REQUESTS_PER_BATCH,
    engine: Sequence[str] = DEFAULT_BATCH_ENGINE,
) -> dict[str, str]:
    """Submit ``tokens`` in batch mode and poll to completion; return token->translit.

    Resumable: if a batch is already in flight for ``checkpoint_path`` this skips
    submission and resumes polling. Tokens whose group output was malformed are
    requeued one-per-request (up to ``requeue_passes``). If ``max_wait`` elapses with
    batches still running, returns what is resolved so far -- rerun later to resume.

    ``engine`` decides how much is answered before anything is submitted. The
    default reads the packaged table first; ``("lookup", "model", "llm")`` also
    decodes locally, and ``("llm",)`` submits everything.
    """
    checkpoint_path = Path(checkpoint_path)
    unique_tokens = [token for token in dict.fromkeys(tokens) if token]
    start = time.time()

    if "llm" not in engine:
        # Handled entirely here, before _make_transliterator. Guarding only the
        # requeue further down still demanded an API key on the way in, so a
        # documented local-only probe failed with "No LLM provider detected".
        # No provider is involved on this path, so the provider, model, key,
        # completion window and batch size have nothing to apply to.
        submit_transliteration_batches(  # preen: allow-dropped-arg
            unique_tokens,
            source_lang,
            target_lang,
            checkpoint_path=checkpoint_path,
            group_size=group_size,
            use_few_shot=use_few_shot,
            temperature=temperature,
            engine=engine,
        )
        return _load_resolved(checkpoint_path)

    transliterator = _make_transliterator(
        source_lang, target_lang, provider, model, api_key, temperature
    )

    def _poll_to_done() -> tuple[dict[str, str], bool]:
        while True:
            done, resolved = collect_transliteration_batches(
                checkpoint_path, transliterator=transliterator, api_key=api_key
            )
            if done:
                return resolved, True
            if max_wait is not None and time.time() - start > max_wait:
                return resolved, False
            time.sleep(poll_interval)

    def _resolve_tail(tokens: list[str]) -> dict[str, str]:
        """Run the backends that sit *after* ``llm``, and record what they answer.

        ``_resolve_locally`` only runs the prefix before ``llm``, so a chain like
        ``("lookup", "llm", "model")`` never reached ``model`` at all: the words
        the provider missed went straight back to the provider. Free backends
        the caller explicitly asked for must be tried before anything is paid
        for a second time.
        """
        tail = tuple(itertools.dropwhile(lambda name: name != "llm", engine))[1:]
        if not tokens or not tail:
            return {}
        # The trailing "llm" is a sentinel, never run: _resolve_locally executes
        # the prefix before it, and here that prefix is exactly the tail.
        answered = _resolve_locally(tokens, source_lang, target_lang, (*tail, "llm"))
        if answered:
            _append_resolved(checkpoint_path, answered)
        return answered

    existing = _load_state(checkpoint_path)
    if existing is not None:
        _validate_resume(existing, transliterator)
    if existing is None:
        submit_transliteration_batches(
            unique_tokens,
            source_lang,
            target_lang,
            checkpoint_path=checkpoint_path,
            provider=provider,
            model=model,
            api_key=api_key,
            group_size=group_size,
            completion_window=completion_window,
            use_few_shot=use_few_shot,
            temperature=temperature,
            max_requests_per_batch=max_requests_per_batch,
            engine=engine,
        )
    else:
        # Resuming onto an in-flight batch. Tokens added on this call were never
        # offered to the local prefix, because that runs inside
        # submit_transliteration_batches, which is skipped here -- so they
        # bypassed the free table and went straight to the paid requeue below.
        already = _load_resolved(checkpoint_path)
        fresh = [token for token in unique_tokens if token not in already]
        local = _resolve_locally(fresh, source_lang, target_lang, engine)
        if local:
            _append_resolved(checkpoint_path, local)
            logger.info(
                "Resumed run: %d newly-supplied token(s) answered locally",
                len(local),
            )

    resolved, done = _poll_to_done()
    if not done:
        logger.warning("max_wait exceeded; rerun to resume from checkpoint")
        return resolved
    _clear_submission(checkpoint_path)

    unresolved = [token for token in unique_tokens if token not in resolved]
    resolved.update(_resolve_tail(unresolved))
    unresolved = [token for token in unique_tokens if token not in resolved]
    passes = 0
    # A chain the caller wrote without `llm` must not acquire one here. The
    # requeue below hard-codes engine=("llm",) -- correct when the caller asked
    # for a provider, a surprise bill when they asked engine=("lookup",) and the
    # table simply had no entry.
    if "llm" not in engine:
        requeue_passes = 0
    while unresolved and passes < requeue_passes:
        passes += 1
        logger.info(
            "Requeueing %d unresolved token(s) at group_size=1 (pass %d/%d)",
            len(unresolved),
            passes,
            requeue_passes,
        )
        submit_transliteration_batches(
            unresolved,
            source_lang,
            target_lang,
            checkpoint_path=checkpoint_path,
            provider=provider,
            model=model,
            api_key=api_key,
            group_size=1,
            completion_window=completion_window,
            use_few_shot=use_few_shot,
            temperature=temperature,
            max_requests_per_batch=max_requests_per_batch,
            # These already went through the local backends and were declined;
            # rerunning them would find nothing.
            engine=("llm",),
        )
        resolved, done = _poll_to_done()
        if not done:
            logger.warning("max_wait exceeded during requeue; rerun to resume")
            return resolved
        _clear_submission(checkpoint_path)
        unresolved = [token for token in unique_tokens if token not in resolved]
        resolved.update(_resolve_tail(unresolved))
        unresolved = [token for token in unique_tokens if token not in resolved]

    if unresolved:
        logger.warning(
            "%d token(s) unresolved after %d requeue pass(es): %s",
            len(unresolved),
            requeue_passes,
            unresolved[:10],
        )
    return resolved
