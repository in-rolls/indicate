"""Batch-mode LLM transliteration via provider Batch APIs (LiteLLM).

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

Provider support is LiteLLM's batch support: **openai, azure, vertex_ai, bedrock,
vllm** -- *not* native Anthropic. To use Claude in batch mode, go through Bedrock
(``provider="bedrock"``, model ``"anthropic.claude-sonnet-4-6"``) or add a native
Anthropic ``messages.batches`` adapter later. The default provider is ``openai``.

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
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import litellm

from .llm_indic import IndicLLMTransliterator
from .logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger()

DEFAULT_GROUP_SIZE = 25
DEFAULT_MAX_REQUESTS_PER_BATCH = 50_000
BATCH_ENDPOINT = "/v1/chat/completions"

#: Answer from the table first, submit only what it declines.
DEFAULT_BATCH_ENGINE: tuple[str, ...] = ("lookup", "llm")


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
@dataclass
class BatchJob:
    """One submitted batch (a provider batch id + the tokens it covers)."""

    batch_id: str
    input_file_id: str
    custom_id_to_tokens: dict[str, list[str]]
    status: str = "submitted"  # "submitted" | "done"
    output_file_id: str | None = None


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


def _state_path(checkpoint_path: Path) -> Path:
    return Path(str(checkpoint_path) + ".batchstate.json")


def _save_state(checkpoint_path: Path, state: BatchState) -> None:
    data = asdict(state)
    _state_path(checkpoint_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_state(checkpoint_path: Path) -> BatchState | None:
    path = _state_path(checkpoint_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    jobs = [BatchJob(**job) for job in data.get("jobs", [])]
    return BatchState(
        provider=data["provider"],
        model=data["model"],
        source_lang=data["source_lang"],
        target_lang=data["target_lang"],
        group_size=data["group_size"],
        temperature=data["temperature"],
        use_few_shot=data["use_few_shot"],
        jobs=jobs,
        submitted_at=data.get("submitted_at"),
    )


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


def _content_to_text(content) -> str:
    """Normalise litellm.file_content's return value to a UTF-8 string."""
    if isinstance(content, (bytes, bytearray)):
        return bytes(content).decode("utf-8")
    if isinstance(content, str):
        return content
    for attr in ("text", "content"):
        value = getattr(content, attr, None)
        if isinstance(value, (bytes, bytearray)):
            return bytes(value).decode("utf-8")
        if isinstance(value, str):
            return value
    return str(content)


def _parse_output_jsonl(content) -> dict[str, str]:
    """Map each request's ``custom_id`` to the model's text output.

    Output lines follow the OpenAI batch schema (``custom_id``,
    ``response.body.choices[0].message.content``, ``error``).
    """
    results: dict[str, str] = {}
    for line in _content_to_text(content).splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        custom_id = record.get("custom_id")
        if not custom_id or record.get("error"):
            continue
        body = (record.get("response") or {}).get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            continue
        message = choices[0].get("message") or {}
        results[custom_id] = message.get("content") or ""
    return results


# --------------------------------------------------------------------------- #
# Provider dispatch: OpenAI (litellm) vs native Gemini batch
# --------------------------------------------------------------------------- #
_GEMINI_PROVIDERS = {"gemini", "google", "google_ai_studio"}


def _gemini_client():
    from google import genai

    return genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from env


def _gemini_model_id(model: str) -> str:
    """Strip a litellm-style 'gemini/' prefix for the native google-genai SDK."""
    return model.split("/", 1)[1] if model.startswith("gemini/") else model


def _gemini_request_line(
    custom_id: str, messages: list[dict], temperature: float
) -> dict:
    """Convert OpenAI-style [system,user] messages to a Gemini batch JSONL line.

    No max_output_tokens cap: gemini-2.5-flash spends output tokens on "thinking", so a
    tight cap truncates the answer. thinking_budget=0 disables reasoning for this simple
    task -- cheaper, faster, and avoids truncation.
    """
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    return {
        "key": custom_id,
        "request": {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generation_config": {
                "temperature": temperature,
                "thinking_config": {"thinking_budget": 0},
            },
        },
    }


def _parse_gemini_jsonl(content) -> dict[str, str]:
    """Map each request's ``key`` to the model text (google-genai batch output)."""
    results: dict[str, str] = {}
    for line in _content_to_text(content).splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        key = record.get("key")
        if not key or record.get("error"):
            continue
        candidates = (record.get("response") or {}).get("candidates") or []
        if not candidates:
            continue
        parts = (candidates[0].get("content") or {}).get("parts") or []
        results[key] = "".join(p.get("text", "") for p in parts)
    return results


def _write_jsonl(records: list[dict]) -> Path:
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 - delete=False, path returned
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for record in records:
        tmp.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.close()
    return Path(tmp.name)


def _submit_openai_jobs(
    state,
    transliterator,
    model,
    custom_id_to_tokens,
    examples,
    temperature,
    completion_window,
    max_requests_per_batch,
):
    requests = [
        {
            "custom_id": cid,
            "method": "POST",
            "url": BATCH_ENDPOINT,
            "body": {
                "model": model,
                "messages": transliterator.build_group_messages(group, examples),
                "temperature": temperature,
                "max_completion_tokens": transliterator.default_max_tokens_for(group),
            },
        }
        for cid, group in custom_id_to_tokens.items()
    ]
    for request_chunk in _chunk(requests, max_requests_per_batch):
        path = _write_jsonl(request_chunk)
        try:
            with path.open("rb") as handle:
                # litellm types the sync variants as possibly-Coroutine
                file_obj = cast(
                    "Any",
                    litellm.create_file(
                        file=handle, purpose="batch", custom_llm_provider=state.provider
                    ),
                )
            batch = cast(
                "Any",
                litellm.create_batch(
                    completion_window=completion_window,
                    endpoint=BATCH_ENDPOINT,
                    input_file_id=file_obj.id,
                    custom_llm_provider=state.provider,
                ),
            )
        finally:
            path.unlink()
        chunk_map = {
            r["custom_id"]: custom_id_to_tokens[r["custom_id"]] for r in request_chunk
        }
        state.jobs.append(
            BatchJob(
                batch_id=batch.id,
                input_file_id=file_obj.id,
                custom_id_to_tokens=chunk_map,
            )
        )
        logger.info(
            "Submitted OpenAI batch %s (%d requests)", batch.id, len(request_chunk)
        )


def _submit_gemini_jobs(
    state,
    transliterator,
    model,
    custom_id_to_tokens,
    examples,
    temperature,
    max_requests_per_batch,
):
    from google.genai import types

    client = _gemini_client()
    gmodel = _gemini_model_id(model)
    requests = [
        _gemini_request_line(
            cid,
            transliterator.build_group_messages(group, examples),
            temperature,
        )
        for cid, group in custom_id_to_tokens.items()
    ]
    for request_chunk in _chunk(requests, max_requests_per_batch):
        path = _write_jsonl(request_chunk)
        try:
            uploaded = client.files.upload(
                file=path, config=types.UploadFileConfig(mime_type="jsonl")
            )
            if not uploaded.name:
                raise RuntimeError("Gemini file upload returned no name")
            job = client.batches.create(model=gmodel, src=uploaded.name)
            if not job.name:
                raise RuntimeError("Gemini batch create returned no name")
        finally:
            path.unlink()
        chunk_map = {r["key"]: custom_id_to_tokens[r["key"]] for r in request_chunk}
        state.jobs.append(
            BatchJob(
                batch_id=job.name,
                input_file_id=uploaded.name,
                custom_id_to_tokens=chunk_map,
            )
        )
        logger.info(
            "Submitted Gemini batch %s (%d requests)", job.name, len(request_chunk)
        )


def _poll_job(job: BatchJob, provider: str) -> tuple[str, dict[str, str]]:
    """Poll one job.

    Returns (status, by_custom_id); status in completed/running/failed.
    """
    if provider in _GEMINI_PROVIDERS:
        client = _gemini_client()  # keep a ref for the whole call (httpx lifecycle)
        info = client.batches.get(name=job.batch_id)
        state_name = getattr(info.state, "name", str(info.state))
        if state_name == "JOB_STATE_SUCCEEDED":
            dest = getattr(info, "dest", None)
            fname = getattr(dest, "file_name", None) if dest else None
            if not fname:
                return "failed", {}
            return "completed", _parse_gemini_jsonl(client.files.download(file=fname))
        if state_name in (
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        ):
            return "failed", {}
        return "running", {}

    retrieved = litellm.retrieve_batch(
        batch_id=job.batch_id, custom_llm_provider=cast("Any", provider)
    )
    status = getattr(retrieved, "status", None)
    output_file_id = getattr(retrieved, "output_file_id", None)
    if status == "completed" and output_file_id:
        job.output_file_id = output_file_id
        return "completed", _parse_output_jsonl(
            litellm.file_content(file_id=output_file_id, custom_llm_provider=provider)
        )
    if status in ("failed", "cancelled", "expired"):
        return "failed", {}
    return "running", {}


# --------------------------------------------------------------------------- #
# Submit
# --------------------------------------------------------------------------- #
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
    completion_window: str = "24h",
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

    state = _load_state(checkpoint_path) or BatchState(
        provider=provider,
        model=model,
        source_lang=source_lang,
        target_lang=target_lang,
        group_size=group_size,
        temperature=temperature,
        use_few_shot=use_few_shot,
    )

    if not todo:
        _save_state(checkpoint_path, state)
        return state

    examples = transliterator.generate_few_shot_examples() if use_few_shot else []

    # One request per group of tokens; custom_id maps back to the group.
    custom_id_to_tokens: dict[str, list[str]] = {}
    for index, group in enumerate(_chunk(todo, group_size)):
        custom_id_to_tokens[f"grp-{index}"] = group

    if provider in _GEMINI_PROVIDERS:
        _submit_gemini_jobs(
            state,
            transliterator,
            model,
            custom_id_to_tokens,
            examples,
            temperature,
            max_requests_per_batch,
        )
    else:
        _submit_openai_jobs(
            state,
            transliterator,
            model,
            custom_id_to_tokens,
            examples,
            temperature,
            completion_window,
            max_requests_per_batch,
        )

    if state.submitted_at is None:
        state.submitted_at = time.time()
    _save_state(checkpoint_path, state)
    return state


# --------------------------------------------------------------------------- #
# Collect
# --------------------------------------------------------------------------- #
def collect_transliteration_batches(
    checkpoint_path: str | Path,
    *,
    transliterator: IndicLLMTransliterator | None = None,
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
            None,
            state.temperature,
        )

    all_done = True
    newly_resolved: dict[str, str] = {}
    for job in state.jobs:
        if job.status == "done":
            continue
        status, by_custom_id = _poll_job(job, state.provider)

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
    completion_window: str = "24h",
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
        submit_transliteration_batches(
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
                checkpoint_path, transliterator=transliterator
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

    if _load_state(checkpoint_path) is None:
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
    _state_path(checkpoint_path).unlink(missing_ok=True)

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
        _state_path(checkpoint_path).unlink(missing_ok=True)
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
