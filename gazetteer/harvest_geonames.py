"""Harvest India place-name pairs from the GeoNames dumps (CC-BY 4.0).

GeoNames is the second permissively-licensed source of Indic-script place names,
and that is exactly why it matters: ``high`` confidence requires two *independent*
trust groups, so until now almost every place name in the corpus was stuck at
``medium`` on Wikidata's word alone.

Two files, both India-scoped by construction -- as with the Wikidata harvester,
the scope is a correctness condition, not a speed optimization. A worldwide pull
would offer Devanagari labels for foreign places and invite exactly the
translation contamination the India filter exists to prevent.

``IN.zip`` -> ``IN.txt``
    The main table. Column 3, ``asciiname``, is the Latin fallback.
``alternatenames/IN.zip`` -> ``IN.txt``
    ``alternateNamesV2`` rows: one name per line, tagged with an ISO 639 code.

Pairing is by ``geonameid``: the ``hi`` (or ``pa``) alternate names are the
native side, and the ``en`` alternate names are the Latin side, falling back to
``asciiname`` when the entity has no English alternate. Multi-token names are
decomposed positionally, the same way Wikidata labels are, because that is how
``नई दिल्ली``/``New Delhi`` yields the particle ``नई``->``new``.

Rows flagged ``isHistoric`` are skipped -- a superseded name is a real
attestation of something, but not of how the place is romanized now. So are the
non-language ``isolanguage`` codes (``post``, ``iata``, ``link``, ``wkdt``,
``abbr`` and friends), which are identifiers rather than names.

Run::

    uv run python -m gazetteer.harvest_geonames --lang hindi

The dumps are downloaded on first use into ``gazetteer/build/geonames/`` and
reused after that; pass ``--input`` to point at copies you already have.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from gazetteer.align import align_tokens
from gazetteer.records import CandidateRow, aggregate, write_rows

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "gazetteer" / "build" / "src"
CACHE_DIR = REPO_ROOT / "gazetteer" / "build" / "geonames"

BASE_URL = "https://download.geonames.org/export/dump"
USER_AGENT = "indicate-gazetteer/0.1 (https://github.com/in-rolls/indicate)"

#: ISO 639 codes GeoNames tags each language's names with.
LANG_CODES = {"hindi": "hi", "punjabi": "pa"}

#: ``isolanguage`` values that are identifiers or codes rather than names.
NOT_A_LANGUAGE = frozenset(
    {"post", "iata", "icao", "faac", "fr_1793", "abbr", "link", "wkdt", "unlc", "phon"}
)

#: Column order of the main geoname table, from the dump's readme.txt.
GEONAME_COLUMNS = 19
_ID, _NAME, _ASCIINAME = 0, 1, 2

#: Column order of alternateNamesV2, from the dump's readme.txt.
_ALT_GEONAMEID, _ALT_ISOLANG, _ALT_NAME, _ALT_HISTORIC = 1, 2, 3, 7


def _download(url: str, dest: Path, member: str) -> Path:
    """Fetch a zip archive and extract one named member.

    Args:
        url: Archive URL.
        dest: Path the extracted text file should end up at.
        member: Exact archive member to extract. Named rather than guessed:
            every GeoNames zip also carries a ``readme.txt``, so taking the
            first ``.txt`` silently yields the readme and a harvest of zero.

    Returns:
        ``dest``.

    Raises:
        KeyError: If the archive does not contain ``member``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if member not in archive.namelist():
            raise KeyError(f"{url} has no member {member!r}: {archive.namelist()}")
        dest.write_bytes(archive.read(member))
    return dest


def ensure_dumps(cache_dir: Path = CACHE_DIR) -> tuple[Path, Path]:
    """Return paths to the two India dumps, downloading them if absent.

    Args:
        cache_dir: Directory to cache the extracted files in.

    Returns:
        ``(main_table, alternate_names)``.
    """
    main = cache_dir / "IN.txt"
    alternates = cache_dir / "alternateNames.IN.txt"
    if not main.is_file():
        print(f"downloading {BASE_URL}/IN.zip", file=sys.stderr)
        _download(f"{BASE_URL}/IN.zip", main, "IN.txt")
    if not alternates.is_file():
        print(f"downloading {BASE_URL}/alternatenames/IN.zip", file=sys.stderr)
        _download(f"{BASE_URL}/alternatenames/IN.zip", alternates, "IN.txt")
    return main, alternates


def _rows(path: Path):
    """Yield split lines from a GeoNames TSV, which has no header or quoting."""
    with path.open(encoding="utf-8", newline="") as handle:
        # QUOTE_NONE: place names legitimately contain double quotes, and the
        # dumps do not quote fields at all.
        yield from csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)


def load_ascii_names(main: Path) -> dict[str, str]:
    """Map geonameid to its ASCII name, the Latin fallback.

    Args:
        main: Path to the main table (``IN.txt``).

    Returns:
        geonameid to ``asciiname``.
    """
    names: dict[str, str] = {}
    for record in _rows(main):
        if len(record) >= GEONAME_COLUMNS and record[_ASCIINAME]:
            names[record[_ID]] = record[_ASCIINAME]
    return names


def load_alternates(alternates: Path) -> dict[str, dict[str, list[str]]]:
    """Group alternate names by geonameid and language.

    Args:
        alternates: Path to the alternateNamesV2 file.

    Returns:
        geonameid to language code to names, historic and non-language rows
        already removed.
    """
    grouped: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in _rows(alternates):
        if len(record) <= _ALT_HISTORIC:
            continue
        code = record[_ALT_ISOLANG]
        name = record[_ALT_NAME]
        historic = record[_ALT_HISTORIC] == "1"
        if not code or not name or historic or code in NOT_A_LANGUAGE:
            continue
        grouped[record[_ALT_GEONAMEID]][code].append(name)
    return {gid: dict(langs) for gid, langs in grouped.items()}


def build_rows(
    ascii_names: dict[str, str],
    alternates: dict[str, dict[str, list[str]]],
    code: str,
) -> list[CandidateRow]:
    """Pair native-script names with Latin names by geonameid.

    Args:
        ascii_names: geonameid to ``asciiname``.
        alternates: geonameid to language code to names.
        code: ISO 639 code for the native language, e.g. ``"hi"``.

    Returns:
        One row per aligned token pair.
    """
    rows: list[CandidateRow] = []
    for geonameid, by_lang in alternates.items():
        natives = by_lang.get(code)
        if not natives:
            continue
        # Every English alternate is an attestation, not just the first; a place
        # with both "Varanasi" and "Banaras" genuinely has two romanizations.
        latins = by_lang.get("en") or []
        fallback = ascii_names.get(geonameid)
        if not latins and fallback:
            latins = [fallback]
        for native in natives:
            for latin in latins:
                for native_token, latin_token in align_tokens(native, latin):
                    try:
                        rows.append(
                            CandidateRow(
                                native=native_token,
                                latin=latin_token,
                                source="geonames",
                                entity_type="geo",
                                weight=1.0,
                                ref=geonameid,
                            )
                        )
                    except ValueError:
                        continue  # e.g. a Latin-only "native" token
    return rows


def harvest(lang: str, *, cache_dir: Path = CACHE_DIR) -> list[CandidateRow]:
    """Harvest GeoNames for one language.

    Args:
        lang: Corpus language name, a key of :data:`LANG_CODES`.
        cache_dir: Where the dumps live or should be downloaded to.

    Returns:
        Deduplicated, attestation-weighted candidate rows.
    """
    main, alternates_path = ensure_dumps(cache_dir)
    ascii_names = load_ascii_names(main)
    alternates = load_alternates(alternates_path)
    return aggregate(build_rows(ascii_names, alternates, LANG_CODES[lang]))


def main(argv: list[str] | None = None) -> int:
    """Harvest GeoNames for one language and write the stage-2 TSV.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True, choices=sorted(LANG_CODES))
    parser.add_argument("--input", type=Path, default=CACHE_DIR)
    parser.add_argument("--out", type=Path, default=BUILD_DIR)
    args = parser.parse_args(argv)

    rows = harvest(args.lang, cache_dir=args.input)
    path = args.out / f"geonames.{args.lang}.tsv"
    count = write_rows(path, rows)
    print(f"geonames/{args.lang}: {count:,} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
