"""Regression tests for source preservation and paid-request controls."""

import csv
import gzip
import json
import sys
from types import SimpleNamespace

import pytest

from training import harvest_kannada as harvest


@pytest.mark.parametrize(
    ("native", "roman", "accepted"),
    [
        ("ವೆಂಕಟರವಣಪ್ಪ", "venkatravanappa", False),
        ("ಅಶ್ವತ್ಥಪ್ಪ", "ashwathappa", False),
        ("ಜವರಶೆಟ್ಟಿ", "javashetti", False),
        ("ಶೆಡ್ತಿ", "shedti", True),
        ("ಲಕ್ಪ್ಮೀ", "lakpmi", True),
        ("ಚಲುವೇಗೌಡ", "chaluve gowda", False),
        ("ಮುನಿಕೃಷ್ಣಪ್ಪ", "munikrishnappa", True),
        ("್ಸ", "ABSTAIN", False),
        ("ಜವರಮ್ಮ", "javaramma", True),
        ("ರಾಮ", "rana", False),
        ("ರಾಮ", "rama", True),
        ("ಖಾನ", "kana", False),
        ("ಖಾನ", "khana", True),
        ("ತಾಯಿ", "thayi", False),
        ("ತಾಯಿ", "tayi", True),
        ("ಹರಿ", "ari", False),
        ("ಹರಿ", "hari", True),
        ("ನಂಜಮ್ಮ", "nanjamma", True),
    ],
)
def test_preservation(native, roman, accepted):
    result, _ = harvest.validate(native, roman)
    assert (result is not None) == accepted


@pytest.fixture
def inventory(tmp_path):
    with gzip.open(tmp_path / "missing_tokens_by_frequency.csv.gz", "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["kannada", "occurrences"])
        for i in range(60):
            writer.writerow(["ರ" + chr(0xC95 + i // 25) + chr(0xC95 + i % 25), 100 - i])
    return tmp_path


def fake_completion(**kwargs):
    lines = kwargs["messages"][-1]["content"].splitlines()[1:]
    data = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": "\n".join(
                        f"{i}. ABSTAIN" for i in range(1, len(lines) + 1)
                    )
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 100},
    }
    return SimpleNamespace(model_dump=lambda: data)


def test_stage_cap_resume_and_unresolved_attempt(inventory, monkeypatch):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return fake_completion(**kwargs)

    monkeypatch.setattr(harvest, "completion", completion)
    monkeypatch.setattr(harvest.getpass, "getpass", lambda _: "test-key")
    argv = ["harvest", "--root", str(inventory), "--limit-requests", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    harvest.main()
    assert len(calls) == 1
    harvest.main()
    assert len(calls) == 1
    out = inventory / "muse_spark_harvest"
    assert len(list(out.glob("response_*.json"))) == 1
    audit = json.loads((out / "audit.json").read_text())
    assert audit["items"] == 25
    assert audit["accepted"] == 0
    assert audit["complete"] is False
    argv[-1] = "2"
    (out / "attempt_0001.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Unresolved attempt"):
        harvest.main()
    assert len(calls) == 1


def test_dry_run_does_not_create_job(inventory, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["harvest", "--root", str(inventory), "--limit-requests", "1", "--dry-run"],
    )
    harvest.main()
    assert not (inventory / "muse_spark_harvest").exists()


def test_cost_bound_stops_before_submission(inventory, monkeypatch):
    monkeypatch.setattr(harvest, "MAX_OUTPUT", 100_000_000)
    with pytest.raises(ValueError, match="Maximum cost"):
        harvest.make_plan(inventory)


@pytest.mark.parametrize("problem", ["length", "numbering", "reserve"])
def test_response_gates(problem):
    response = fake_completion(messages=[{"content": "header\n1. ರಾಮ"}]).model_dump()
    record = {
        "request": {"tokens": ["ರಾಮ"]},
        "response": response,
        "parameters": {"max_completion_tokens": 3072},
    }
    if problem == "length":
        response["choices"][0]["finish_reason"] = "length"
    elif problem == "numbering":
        response["choices"][0]["message"]["content"] = "2. rama"
    else:
        response["usage"]["completion_tokens"] = 2500
    with pytest.raises(ValueError):
        harvest.parse_response(record)


def test_reviewed_suffix_preserves_original_and_rejects_invention():
    import hashlib

    content = "1. partial\n1. Rama"
    record = {
        "request": {"tokens": ["ರಾಮ"]},
        "response": {
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {"completion_tokens": 100},
        },
        "parameters": {"max_completion_tokens": 3072},
        "reviewed_suffix": {
            "original_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "content": "1. Rama",
            "reason": "Repeated prefix; exact complete final numbered suffix reviewed.",
        },
    }
    assert harvest.parse_response(record) == [("ರಾಮ", "rama", None)]
    assert record["response"]["choices"][0]["message"]["content"] == content
    record["reviewed_suffix"]["content"] = "1. Rana"
    with pytest.raises(ValueError, match="does not match"):
        harvest.parse_response(record)
    record["reviewed_suffix"]["content"] = "1. Rama"
    record["reviewed_suffix"]["original_sha256"] = "changed"
    with pytest.raises(ValueError, match="does not match"):
        harvest.parse_response(record)


@pytest.mark.parametrize(
    ("native", "latin", "expected"),
    [
        ("ಅಮಾಸೇಗೌಡ", "amasegouda", "amasegowda"),
        ("ಅಮಾಸೇಗೌಡ", "amasegauda", "amasegowda"),
        ("ಅಮಾಸೇಗೌಡ", "amasegowda", "amasegowda"),
        ("ರಾಮ", "gauda", "gauda"),
    ],
)
def test_source_conditioned_suffix(native, latin, expected):
    from training.merge_kannada_harvest import canonicalize

    assert canonicalize(native, latin) == expected


def test_merge_recovers_unusable_entries_and_preserves_valid_pairs(
    tmp_path, monkeypatch
):
    from training.merge_kannada_harvest import main

    corpus = tmp_path / "corpus.csv.gz"
    with gzip.open(corpus, "wt") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kannada", "english"])
        writer.writerows(
            [
                ("ದೊಡ್ಡಮನಿ", "dodda mani"),
                ("ನಾರಾಯಣಗೌಡ", "narayana gowda"),
                ("ಡಾಕಪ್ಪ", "dr kappa"),
                ("ರಾಮ", "rama"),
            ]
        )
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}")
    root = tmp_path / "muse_spark_harvest"
    root.mkdir()
    (root / "audit.json").write_text(json.dumps({"complete": True, "items": 5}))
    pairs = [
        ("ದೊಡ್ಡಮನಿ", "doddamani"),
        ("ನಾರಾಯಣಗೌಡ", "narayanagouda"),
        ("ಡಾಕಪ್ಪ", "dakappa"),
        ("ರಾಮ", "ramaa"),
        ("ಗೌಡ", "gauda"),
    ]
    (root / "validated_tokens.jsonl").write_text(
        "\n".join(json.dumps({"kannada": n, "english": e}) for n, e in pairs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge",
            "--root",
            str(tmp_path),
            "--corpus",
            str(corpus),
            "--provenance",
            str(provenance),
        ],
    )
    main()
    with gzip.open(corpus, "rt") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["kannada"], r["english"]) for r in rows] == [
        ("ದೊಡ್ಡಮನಿ", "doddamani"),
        ("ನಾರಾಯಣಗೌಡ", "narayanagowda"),
        ("ಡಾಕಪ್ಪ", "dakappa"),
        ("ರಾಮ", "rama"),
        ("ಗೌಡ", "gowda"),
    ]
    audit = json.loads((root / "merge_audit.json").read_text())
    assert audit["added_rows"] == 1
    assert audit["existing_native_keys_skipped"] == 1
    assert len(audit["replaced_unusable_entries"]) == 3
    before = corpus.read_bytes()
    with pytest.raises(ValueError, match="already merged"):
        main()
    assert corpus.read_bytes() == before


def test_merge_rejects_duplicate_candidate_keys(tmp_path, monkeypatch):
    from training.merge_kannada_harvest import main

    corpus = tmp_path / "corpus.csv.gz"
    with gzip.open(corpus, "wt") as handle:
        handle.write("kannada,english\nರಾಮ,rama\n")
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}")
    root = tmp_path / "muse_spark_harvest"
    root.mkdir()
    (root / "audit.json").write_text(json.dumps({"complete": True, "items": 2}))
    (root / "validated_tokens.jsonl").write_text(
        (json.dumps({"kannada": "ರಾಮ", "english": "rama"}) + "\n") * 2
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge",
            "--root",
            str(tmp_path),
            "--corpus",
            str(corpus),
            "--provenance",
            str(provenance),
        ],
    )
    before = corpus.read_bytes()
    with pytest.raises(ValueError, match="duplicate native keys"):
        main()
    assert corpus.read_bytes() == before
