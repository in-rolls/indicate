"""Stage 1: mine token frequency from real corpora to produce the key list.

Run::

    uv run --group train python -m gazetteer.mine_frequency --lang punjabi
    uv run --group train python -m gazetteer.mine_frequency --lang hindi

Writes ``gazetteer/build/freq/<lang>.tsv`` (the key list) and
``<lang>.coverage.json`` (the cut points that decide how big the trunk needs to
be). Only counts are taken from restricted corpora -- romanizations come from
redistributable sources in stage 2 -- so the output of this stage is aggregate
and non-identifying.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from gazetteer.frequency import DEFAULT_MARKS, FrequencyTable, iter_tokens

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "gazetteer" / "build" / "freq"

#: Punjab electoral-roll columns and the entity class each contributes.
#: ``training/extract_punjabi.py`` pools these untagged, losing the distinction
#: between a surname and a village name; the gazetteer needs it kept.
ROLL_FIELDS = {
    "elector_name": "person",
    "father_or_husband_name": "person",
    "main_town": "geo",
    "police_station": "geo",
    "mandal": "geo",
    "revenue_division": "geo",
    "district": "geo",
    "ac_name": "geo",
    "parl_constituency": "geo",
    "polling_station_name": "geo",
    "polling_station_address": "geo",
}

AFFIDAVIT_FIELDS = {
    "name_hindi": "person",
    "fathers_or_husbands_name_hindi": "person",
}


def mine_punjabi(path: Path, max_rows: int | None = None) -> FrequencyTable:
    """Mine Gurmukhi token frequency from the Punjab electoral-roll parquet.

    Args:
        path: Path to the roll parquet.
        max_rows: Stop after roughly this many rows; ``None`` reads all 19.1M.

    Returns:
        The populated frequency table.

    Raises:
        SystemExit: If pyarrow is unavailable.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "pyarrow is required: uv run --group train python -m "
            "gazetteer.mine_frequency ..."
        ) from None

    table = FrequencyTable()
    columns = list(ROLL_FIELDS)
    seen = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=200_000, columns=columns):
        data = batch.to_pydict()
        for column, entity in ROLL_FIELDS.items():
            for value in data[column]:
                table.add_all(iter_tokens(value), entity)
        seen += batch.num_rows
        print(f"  ...{seen:,} rows", file=sys.stderr)
        if max_rows is not None and seen >= max_rows:
            break
    return table


def mine_hindi(repo_root: Path) -> FrequencyTable:
    """Mine Devanagari token frequency from the repo's Hindi name corpora.

    Args:
        repo_root: Repository root holding ``data/``.

    Returns:
        The populated frequency table.
    """
    table = FrequencyTable()

    affidavits = repo_root / "data" / "affidavits.csv"
    if affidavits.is_file():
        with affidavits.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for column, entity in AFFIDAVIT_FIELDS.items():
                    table.add_all(iter_tokens(row.get(column)), entity)

    players = repo_root / "data" / "players_with_hindi_names.json"
    if players.is_file():
        records = json.loads(players.read_text(encoding="utf-8"))
        for record in records:
            for column in ("hindi_name", "hindi_long_name"):
                table.add_all(iter_tokens(record.get(column)), "person")

    return table


def write_outputs(table: FrequencyTable, lang: str, out_dir: Path) -> Path:
    """Write the key list and its coverage report.

    Args:
        table: Mined frequency table.
        lang: Language name used in the filenames.
        out_dir: Directory to write into; created if absent.

    Returns:
        Path of the written TSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = table.rows()

    tsv_path = out_dir / f"{lang}.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["key", "count", "surfaces", "entities"])
        for row in rows:
            writer.writerow(
                [
                    row.key,
                    row.count,
                    "|".join(f"{s}:{c}" for s, c in row.surfaces.items()),
                    "|".join(f"{e}:{c}" for e, c in row.entities.items()),
                ]
            )

    coverage = table.coverage()
    (out_dir / f"{lang}.coverage.json").write_text(
        json.dumps(
            {
                "lang": lang,
                "types": len(rows),
                "tokens": table.total_tokens,
                "cutpoints": {str(m): coverage[m] for m in DEFAULT_MARKS},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return tsv_path


def main(argv: list[str] | None = None) -> int:
    """Mine frequency for one language and write the stage-1 outputs.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True, choices=("punjabi", "hindi"))
    parser.add_argument(
        "--roll",
        type=Path,
        default=REPO_ROOT / "data" / "punjab_transliteration_subset.parquet",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--out", type=Path, default=BUILD_DIR)
    args = parser.parse_args(argv)

    if args.lang == "punjabi":
        if not args.roll.is_file():
            print(f"missing roll parquet: {args.roll}", file=sys.stderr)
            return 1
        table = mine_punjabi(args.roll, args.max_rows)
    else:
        table = mine_hindi(REPO_ROOT)

    if not table.total_tokens:
        print(f"no tokens mined for {args.lang}", file=sys.stderr)
        return 1

    path = write_outputs(table, args.lang, args.out)
    coverage = table.coverage()
    print(f"{args.lang}: {len(table.rows()):,} types, {table.total_tokens:,} tokens")
    for mark in DEFAULT_MARKS:
        print(f"  {mark:>6.1%} of tokens <- top {coverage[mark]:,} keys")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
