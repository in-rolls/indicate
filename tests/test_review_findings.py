"""Regressions for the defects an independent review found before 0.8.0.

Every test here failed when it was written. They are grouped in one file
because they share a cause worth naming: each is a path that only opens when an
artifact is *absent*, and the machine the suite was written on had every
artifact present. The batch pipeline aborting without a lookup table, a build
failure turning into a skip, a status that reports ``ready`` for half a
download -- none of these can be reached from a fully-provisioned checkout.

The two costly ones were in :mod:`indicate.batch`. Its own tests pass
``engine=("llm",)`` so that the fixtures actually reach the fake provider, and
the one test that exercises the table is marked ``needs_lookup`` and skips
without it. Between them, the default ``("lookup", "llm")`` chain on a machine
with no table -- which is every installed user -- was covered by nothing.
"""

from __future__ import annotations

import pytest

import indicate
import indicate.batch as batch_mod
import indicate.lookup as lookup_mod
from indicate.languages import ON_DEMAND, PAIRS, READY, status
from indicate.transliterator import DECODER_FILE, ENCODER_FILE

from . import conftest

PA = PAIRS[("punjabi", "english")]
SINGH = "ਸਿੰਘ"
KAUR = "ਕੌਰ"


@pytest.fixture
def no_table(monkeypatch: pytest.MonkeyPatch):
    """Force every lookup table to be absent, as on a stock install."""
    monkeypatch.setitem(lookup_mod._CACHE, PA.subdir, None)
    return PA.subdir


def test_batch_falls_through_to_the_provider_when_there_is_no_table(no_table):
    # The default batch chain is ("lookup", "llm"). With no table the lookup
    # backend is *unavailable*, not merely declining, so resolve_words raises.
    # In the batch path that must not be fatal: the llm suffix is precisely the
    # fallback, so the correct answer is "nothing resolved locally".
    resolved = batch_mod._resolve_locally(
        [SINGH, KAUR], "punjabi", "english", ("lookup", "llm")
    )
    assert resolved == {}


def test_a_misspelled_backend_raises_instead_of_submitting_everything(no_table):
    # ("lookpu", "llm") used to be swallowed as "no local backend available",
    # after which every token went to a paid provider. A typo must cost an
    # exception, not money.
    with pytest.raises(ValueError, match="lookpu"):
        batch_mod._resolve_locally(
            [SINGH, KAUR], "punjabi", "english", ("lookpu", "llm")
        )


def test_a_misspelled_backend_is_caught_before_any_submission(
    no_table, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    # Same defect at the public entry point, which is where the money is spent.
    # The key must be set: without one the provider check fires first and this
    # would pass for the wrong reason. The case worth pinning is the expensive
    # one -- a user who *can* be billed, typing the chain wrong.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="lookpu"):
        batch_mod.submit_transliteration_batches(
            [SINGH],
            "punjabi",
            "english",
            checkpoint_path=tmp_path / "unused.jsonl",
            engine=("lookpu", "llm"),
        )


def test_a_chain_without_llm_never_submits_anything(no_table, tmp_path):
    # engine=("lookup",) is the documented "is my corpus already covered?"
    # probe. It used to answer the hits from the table and then submit every
    # miss to a paid provider, which is the opposite of what the chain says.
    # litellm is left unpatched deliberately: reaching it at all is the defect,
    # so any submission attempt fails loudly rather than being absorbed.
    resolved = batch_mod.submit_transliteration_batches(
        [SINGH, "ZZZQQ"],
        "punjabi",
        "english",
        checkpoint_path=tmp_path / "probe.jsonl",
        engine=("lookup",),
    )
    assert not batch_mod._state_path(tmp_path / "probe.jsonl").exists()
    assert resolved.jobs == []


def test_detection_looks_past_a_run_of_blank_inputs():
    # Blank entries are explicitly supported and come back as "". Sampling the
    # first 50 *positions* rather than the first 50 *non-blank* texts meant a
    # batch that opens with 50 blanks had nothing to detect from and raised.
    #
    # The assertion is about *detection*, which needs no artifact, so the test
    # must not need one either: BackendsUnavailableError means the pair was
    # resolved and then nothing could run, which is a pass on a fresh clone.
    # UnsupportedPairError is the failure, and it propagates.
    texts = [""] * 50 + ["नमस्ते"]
    try:
        out = indicate.transliterate_batch(texts)
    except indicate.BackendsUnavailableError:
        return
    assert out[-1] == "namaste"
    assert out[:50] == [""] * 50


def test_a_table_that_is_not_utf8_disables_itself_instead_of_raising(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fresh_caches
):
    # fresh_caches, because loading a table caches the result under the real
    # subdir key. Without it this test leaves _CACHE[punjabi] = None behind and
    # every later test that expects a working table fails -- which is exactly
    # what it did, and only in whole-file runs.
    # Valid gzip, invalid UTF-8 -- the decode error surfaces during iteration,
    # not at open(), so it sat outside the caught set and escaped
    # LookupBackend.resolve, taking the model fallback down with it.
    import gzip

    bad = tmp_path / "lookup.tsv.gz"
    bad.write_bytes(gzip.compress(b"#normalizer\t1\n\xff\xfe\tsingh\n"))

    lookup_mod.clear_cache()
    monkeypatch.setattr("indicate.lookup.local_data_path", lambda subdir, rel: str(bad))
    assert lookup_mod.Lookup.load(PA.subdir) is None


def test_dry_run_does_not_create_the_output_directory(tmp_path):
    from click.testing import CliRunner

    from indicate.cli import cli

    missing = tmp_path / "no" / "such" / "dir"
    result = CliRunner().invoke(
        cli,
        ["transliterate", SINGH, "--output", str(missing / "out.txt"), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    # "Would write" and then a directory on disk is not a dry run.
    assert not missing.exists()


@pytest.mark.needs_lookup
def test_a_table_hit_keeps_the_punctuation_it_was_given():
    # lookup_key strips edge punctuation and digit prefixes to build the key, so
    # returning the bare stored value deleted them from the output -- on the
    # default chain, because the table answers first. `022-` is a roll serial;
    # losing it silently corrupts the record.
    hit = indicate.transliterate("ਸਿੰਘ", source="punjabi")
    if hit != "singh":
        pytest.skip("punjabi table not built")
    for given, expected in (
        ("ਸਿੰਘ,", "singh,"),
        ("(ਸਿੰਘ)", "(singh)"),
        ("022-ਸਿੰਘ", "022-singh"),
    ):
        assert indicate.transliterate(given, source="punjabi") == expected


def test_dry_run_never_reaches_a_backend(monkeypatch: pytest.MonkeyPatch, tmp_path):
    # --dry-run used to call _run first and check the flag afterwards, so
    # `--engine llm --dry-run` paid for an answer it then discarded, and without
    # --output the flag was ignored altogether.
    from click.testing import CliRunner

    import indicate.api as api_mod
    from indicate.cli import cli

    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("a dry run reached the backend")

    monkeypatch.setattr(api_mod, "transliterate_batch", explode)
    for args in (
        ["transliterate", SINGH, "--dry-run"],
        ["transliterate", SINGH, "--dry-run", "--output", str(tmp_path / "o.txt")],
        ["transliterate", SINGH, "--dry-run", "--engine", "llm"],
    ):
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 0, (args, result.output)
        assert "Would write" in result.output, args


@pytest.mark.needs_lookup
def test_language_aliases_work_on_the_local_batch_path(tmp_path):
    # PAIRS is keyed on canonical names, and the llm path normalizes for free
    # via IndicLLMTransliterator -- so "pa"/"en", which are documented aliases,
    # resolved nothing locally while the *paid* path handled them fine.
    if batch_mod._resolve_locally([SINGH], "punjabi", "english", ("lookup",)) == {}:
        pytest.skip("punjabi table not built")
    assert batch_mod._resolve_locally([SINGH], "pa", "en", ("lookup",)) == {
        SINGH: "singh"
    }
    # An unknown name is a miss, not a crash.
    assert batch_mod._resolve_locally([SINGH], "klingon", "en", ("lookup",)) == {}


@pytest.mark.needs_lookup
def test_the_local_only_driver_needs_no_api_key(tmp_path, monkeypatch):
    # transliterate_tokens_batched called _make_transliterator before reaching
    # the no-llm guard, so a local-only probe died on "No LLM provider
    # detected" before it ever consulted the table.
    # Every one of them, including GEMINI_API_KEY. Leaving one behind lets
    # provider detection succeed and the test passes without exercising the
    # guard at all -- which is exactly what it did on the first attempt.
    for name in conftest._PROVIDER_KEYS:
        monkeypatch.delenv(name, raising=False)
    out = batch_mod.transliterate_tokens_batched(
        [SINGH, "ZZZQQ"],
        "punjabi",
        "english",
        checkpoint_path=tmp_path / "probe.jsonl",
        engine=("lookup",),
    )
    if not out:
        pytest.skip("punjabi table not built")
    assert out == {SINGH: "singh"}


def test_cli_detection_looks_past_leading_blank_lines(runner, text_file, tmp_path):
    # Same defect as the API one, in the other entry point: blank lines are
    # preserved deliberately, so sampling the first 50 *positions* left
    # detection with nothing but blanks.
    from indicate.cli import cli

    src = text_file("\n" * 50 + "ਸਿੰਘ\n")
    out = tmp_path / "o.txt"
    result = runner.invoke(
        cli, ["transliterate", "--input", str(src), "--output", str(out)]
    )
    # Artifact-independent, like its API twin: what must never appear is the
    # detection failure. "nothing could answer" means the pair resolved.
    assert "could not detect" not in result.output, result.output
    if result.exit_code != 0:
        assert "nothing could answer" in result.output, result.output
        return
    assert out.read_text(encoding="utf-8").strip().endswith("singh")


def test_status_is_not_ready_when_only_half_the_weights_are_present(
    monkeypatch: pytest.MonkeyPatch,
):
    # An interrupted download leaves the encoder and no decoder. Reporting
    # `ready` there sends the user to a backend that raises on first use.
    present = {ENCODER_FILE}

    def only_encoder(subdir: str, rel: str) -> str:
        return "/exists" if rel in present else "/definitely/missing"

    monkeypatch.setattr("indicate.resources.local_data_path", only_encoder)
    monkeypatch.setattr("os.path.exists", lambda p: p == "/exists")
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)

    # Not READY. It is ON_DEMAND rather than UNAVAILABLE because the missing
    # half is still fetchable -- which is the honest answer, and the point is
    # that the user is no longer told it will work right now when it will not.
    assert status("punjabi", "english")["model"] == ON_DEMAND

    present.add(DECODER_FILE)
    assert status("punjabi", "english")["model"] == READY
