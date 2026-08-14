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
    texts = [""] * 50 + ["नमस्ते"]
    out = indicate.transliterate_batch(texts)
    assert out[-1] == "namaste"
    assert out[:50] == [""] * 50


def test_a_table_that_is_not_utf8_disables_itself_instead_of_raising(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
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
