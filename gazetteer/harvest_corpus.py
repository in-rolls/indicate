"""Harvest bilingual name pairs from the repo's own permissively-licensed data.

Two corpora qualify for the bundle. ``data/affidavits.csv`` holds candidate
names as declared on election affidavits, in Devanagari and Latin side by side.
``data/players_with_hindi_names.json`` holds editorially-maintained cricketer
names from ESPNCricinfo.

Deliberately excluded: ``data/hindi.csv.gz`` blends these two with IIT Bombay
mined pairs, which are CC-BY-NC, so harvesting it would contaminate a corpus
published under CC-BY-4.0. Reading the two clean sources directly costs nothing
and keeps the licence story checkable. ``data/punjabi.csv.gz`` is excluded for a
different reason -- it is GPT-4o output derived from the restricted roll, and
the shipped model was trained on it, so it is neither redistributable nor
independent.

Run::

    uv run python -m gazetteer.harvest_corpus --lang hindi
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path

from gazetteer.align import align_tokens
from gazetteer.records import CandidateRow, aggregate, write_rows

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "gazetteer" / "build" / "src"

#: Affidavit column pairs: (native column, latin column).
AFFIDAVIT_COLUMNS = (
    ("name_hindi", "name_english"),
    ("fathers_or_husbands_name_hindi", "fathers_or_husbands_name_english"),
)

#: Cricinfo record keys: (native key, latin key).
CRICINFO_COLUMNS = (("hindi_name", "english_name"),)


def _rows(
    labels: Iterable[tuple[str | None, str | None]],
    source: str,
    entity_type: str,
) -> list[CandidateRow]:
    rows: list[CandidateRow] = []
    for native, latin in labels:
        for native_token, romanization in align_tokens(native, latin):
            try:
                rows.append(
                    CandidateRow(
                        native=native_token,
                        latin=romanization,
                        source=source,
                        entity_type=entity_type,
                        weight=1.0,
                    )
                )
            except ValueError:
                continue
    return rows


def harvest_affidavits(path: Path) -> list[CandidateRow]:
    """Harvest bilingual candidate names from the affidavits CSV.

    Args:
        path: Path to ``data/affidavits.csv``.

    Returns:
        Candidate rows, or ``[]`` if the file is absent.
    """
    if not path.is_file():
        return []
    labels: list[tuple[str | None, str | None]] = []
    with path.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            labels.extend(
                (record.get(native), record.get(latin))
                for native, latin in AFFIDAVIT_COLUMNS
            )
    return _rows(labels, "affidavits", "person")


def harvest_cricinfo(path: Path) -> list[CandidateRow]:
    """Harvest bilingual player names from the ESPNCricinfo JSON.

    Args:
        path: Path to ``data/players_with_hindi_names.json``.

    Returns:
        Candidate rows, or ``[]`` if the file is absent.
    """
    if not path.is_file():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    labels: list[tuple[str | None, str | None]] = [
        (record.get(native), record.get(latin))
        for record in records
        for native, latin in CRICINFO_COLUMNS
    ]
    return _rows(labels, "cricinfo", "person")


def main(argv: list[str] | None = None) -> int:
    """Harvest the repo-owned corpora and write stage-2 TSVs.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="hindi", choices=("hindi",))
    parser.add_argument("--out", type=Path, default=BUILD_DIR)
    args = parser.parse_args(argv)

    data = REPO_ROOT / "data"
    for source, rows in (
        ("affidavits", harvest_affidavits(data / "affidavits.csv")),
        ("cricinfo", harvest_cricinfo(data / "players_with_hindi_names.json")),
    ):
        path = args.out / f"{source}.{args.lang}.tsv"
        count = write_rows(path, aggregate(rows))
        print(f"{source}/{args.lang}: {count:,} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
