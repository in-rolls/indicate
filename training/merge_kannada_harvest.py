"""Merge structurally screened Kannada candidates and record their provenance."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

from training.harvest_kannada import validate


def canonicalize(native, latin):
    """Use the majority compound spelling for newly promoted ಗೌಡ values."""
    if native.endswith("ಗೌಡ"):
        for suffix in ("gouda", "gauda"):
            if latin.endswith(suffix):
                return latin[: -len(suffix)] + "gowda"
    return latin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    harvest = args.root / "muse_spark_harvest"
    audit = json.loads((harvest / "audit.json").read_text())
    if not audit["complete"]:
        raise ValueError("Harvest is incomplete")
    provenance = json.loads(args.provenance.read_text())
    if "muse_spark_2017" in provenance:
        raise ValueError("Harvest already merged; verify before rebuilding")
    with gzip.open(args.corpus, "rt") as handle:
        rows = list(csv.DictReader(handle))
    known = {row["kannada"] for row in rows}
    if len(known) != len(rows):
        raise ValueError("Existing corpus has duplicate native keys")
    positions = {row["kannada"]: i for i, row in enumerate(rows)}
    additions, normalized, existing, rejected = [], [], [], []
    replacements = []
    candidates = [
        json.loads(line)
        for line in (harvest / "validated_tokens.jsonl").read_text().splitlines()
    ]
    if len({row["kannada"] for row in candidates}) != len(candidates):
        raise ValueError("Harvest has duplicate native keys")
    if len(candidates) != audit["items"]:
        raise ValueError("Candidate count disagrees with audit")
    for row in candidates:
        native, latin = row["kannada"], row["english"]
        if latin is None:
            continue
        screened, reason = validate(native, latin)
        if screened is None:
            raise ValueError(f"Candidate no longer passes: {reason}")
        canonical = canonicalize(native, latin)
        if validate(native, canonical)[0] is None:
            rejected.append(
                {
                    "kannada": native,
                    "english": latin,
                    "reason": "canonicalization-failed-screen",
                }
            )
            continue
        if native in known:
            previous = rows[positions[native]]["english"]
            if validate(native, previous)[0] is None:
                replacements.append(
                    {
                        "kannada": native,
                        "previous_english": previous,
                        "model_romanization": latin,
                        "english": canonical,
                    }
                )
                rows[positions[native]] = {"kannada": native, "english": canonical}
            else:
                existing.append(native)
            continue
        if canonical != latin:
            normalized.append(
                {
                    "kannada": native,
                    "model_romanization": latin,
                    "corpus_romanization": canonical,
                }
            )
        additions.append({"kannada": native, "english": canonical})
        known.add(native)
    original_sha = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    temporary = args.corpus.with_suffix(".gz.tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped,
        io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle,
    ):
        writer = csv.DictWriter(handle, fieldnames=["kannada", "english"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerows(additions)
    report = {
        "original_sha256": original_sha,
        "original_rows": len(rows),
        "added_rows": len(additions),
        "rows": len(rows) + len(additions),
        "sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
        "harvest_audit": audit,
        "existing_native_keys_skipped": len(existing),
        "replaced_unusable_entries": replacements,
        "post_normalization_rejected": rejected,
        "normalizations": normalized,
        "canonicalization": (
            "Newly promoted values (additions and replacements): native suffix "
            "ಗೌಡ with gauda/gouda "
            "becomes gowda, the majority existing compound spelling "
            "(5946 gowda,1143 gauda,125 gouda). Revalidated."
        ),
        "quality": (
            "Structurally screened LLM spellings; not independently verified "
            "name accuracy. Existing valid pairs unchanged; unusable entries "
            "replaced with screened candidates and individually recorded."
        ),
    }
    (harvest / "merge_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    temporary.replace(args.corpus)
    provenance["muse_spark_2017"] = {
        k: v
        for k, v in report.items()
        if k
        not in (
            "normalizations",
            "post_normalization_rejected",
            "replaced_unusable_entries",
        )
    }
    provenance["rows"] = report["rows"]
    provenance["current_sha256"] = report["sha256"]
    provenance["method"] = (
        "Existing Kannada corpus plus structurally screened "
        "Muse Spark 1.3 Contributor isolated-token harvest."
    )
    args.provenance.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "original_rows",
                    "added_rows",
                    "rows",
                    "sha256",
                    "existing_native_keys_skipped",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
