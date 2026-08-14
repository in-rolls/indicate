"""Harvest India-scoped bilingual labels from Wikidata (CC0).

Wikidata is the highest value-per-effort source: labels in an Indic language and
in English live on the same entity, so the join is free, and CC0 places no
conditions on redistribution.

**The India scope is a correctness condition, not an optimization.** An unscoped
pull of 4,476 Punjabi person labels produced ``ਮਸੀਹ``->``christ`` (a token with
130,862 electoral-roll occurrences), ``ਪਾਲ``->``paul`` (49,929) and
``ਚੱਕ``->``chuck`` (45,518) -- translations and European homographs that would
be published as confident romanizations. Every query here is therefore bound to
``wd:Q668``.

Multi-token labels are decomposed positionally into token pairs. That is the
only way to obtain name *particles*: Wikidata's given-name and family-name items
carry almost no Indic labels (75 for ``hi``, 142 for ``bn``, against 26,946 for
``en``), but full person labels are plentiful.

Run::

    uv run python -m gazetteer.harvest_wikidata --lang punjabi
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from gazetteer.align import align_tokens
from gazetteer.records import CandidateRow, aggregate, write_rows

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "gazetteer" / "build" / "src"

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "indicate-gazetteer/0.1 (https://github.com/in-rolls/indicate)"

#: Wikidata language codes for the languages the corpus covers.
LANG_CODES = {"punjabi": "pa", "hindi": "hi"}

#: Entity class -> the triple pattern that binds it to India. ``wd:Q668`` is
#: India; ``P27`` is citizenship, ``P19`` birthplace, ``P17`` country.
ENTITY_CLASSES = {
    "person": """
      ?item wdt:P31 wd:Q5 .
      { ?item wdt:P27 wd:Q668 } UNION { ?item wdt:P19/wdt:P17 wd:Q668 }
    """,
    "geo": """
      ?item wdt:P17 wd:Q668 .
    """,
}

_QUERY_TEMPLATE = """SELECT DISTINCT ?item ?native ?en WHERE {{
{scope}
  ?item rdfs:label ?native . FILTER(lang(?native) = "{lang}")
  ?item rdfs:label ?en . FILTER(lang(?en) = "en")
}}
ORDER BY ?item
LIMIT {limit} OFFSET {offset}"""
# ORDER BY is load-bearing, not tidiness: SPARQL guarantees no row order without
# it, so paging with LIMIT/OFFSET over an unordered result can return the same
# row on two pages and never return another. That silently changes what the
# corpus contains between runs, which is fatal for something whose whole claim
# is reproducible provenance.


def build_query(lang: str, entity: str, *, limit: int, offset: int) -> str:
    """Build the SPARQL query for one page of one entity class.

    Args:
        lang: Wikidata language code, e.g. ``"pa"``.
        entity: Key of :data:`ENTITY_CLASSES`.
        limit: Page size.
        offset: Page offset.

    Returns:
        The SPARQL query text.

    Raises:
        ValueError: If ``entity`` is not a known class, which would otherwise
            silently produce an unscoped query.
    """
    if entity not in ENTITY_CLASSES:
        raise ValueError(
            f"unknown entity class: {entity!r}; "
            f"expected one of {sorted(ENTITY_CLASSES)}"
        )
    return _QUERY_TEMPLATE.format(
        scope=ENTITY_CLASSES[entity].rstrip(), lang=lang, limit=limit, offset=offset
    )


def rows_from_bindings(
    bindings: Sequence[dict], entity: str, *, weight: float = 1.0
) -> list[CandidateRow]:
    """Convert SPARQL result bindings into candidate rows.

    Args:
        bindings: The ``results.bindings`` list from a WDQS JSON response.
        entity: Entity class to tag the rows with.
        weight: Within-source confidence for these rows.

    Returns:
        One row per aligned token pair. Labels whose token counts disagree, or
        which contain no Indic content, contribute nothing.
    """
    rows: list[CandidateRow] = []
    for binding in bindings:
        native = binding.get("native", {}).get("value")
        english = binding.get("en", {}).get("value")
        qid = binding.get("item", {}).get("value", "").rsplit("/", 1)[-1]
        for native_token, latin in align_tokens(native, english):
            try:
                rows.append(
                    CandidateRow(
                        native=native_token,
                        latin=latin,
                        source="wikidata",
                        entity_type=entity,
                        weight=weight,
                        ref=qid,
                    )
                )
            except ValueError:
                continue  # e.g. a Latin-only "native" token
    return rows


def run_query(query: str, *, timeout: int = 90) -> list[dict]:
    """Execute a SPARQL query against WDQS and return its bindings.

    Args:
        query: SPARQL query text.
        timeout: Socket timeout in seconds.

    Returns:
        The ``results.bindings`` list.
    """
    url = f"{ENDPOINT}?{urllib.parse.urlencode({'query': query, 'format': 'json'})}"
    request = urllib.request.Request(  # noqa: S310 - fixed https endpoint
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.load(response)
    return payload.get("results", {}).get("bindings", [])


def harvest(
    lang: str, *, page_size: int = 5000, max_pages: int = 40
) -> list[CandidateRow]:
    """Harvest every entity class for one language.

    Args:
        lang: Corpus language name, a key of :data:`LANG_CODES`.
        page_size: Rows requested per query.
        max_pages: Safety cap on pages per entity class.

    Returns:
        Deduplicated candidate rows.
    """
    code = LANG_CODES[lang]
    rows: list[CandidateRow] = []
    for entity in ENTITY_CLASSES:
        for page in range(max_pages):
            query = build_query(code, entity, limit=page_size, offset=page * page_size)
            try:
                bindings = run_query(query)
            except Exception as exc:
                # Raise, do not break. Breaking returned whatever had been
                # collected so far, and main() then overwrote a complete TSV
                # with the truncated result and exited 0 -- so a transient WDQS
                # timeout silently shrank the corpus and looked like a success.
                raise RuntimeError(
                    f"{entity} page {page} failed after {len(rows):,} rows: {exc}. "
                    f"Refusing to write a partial harvest over a complete one."
                ) from exc
            rows.extend(rows_from_bindings(bindings, entity))
            print(
                f"  {entity} page {page}: {len(bindings):,} bindings, "
                f"{len(rows):,} rows so far",
                file=sys.stderr,
            )
            if len(bindings) < page_size:
                break
            time.sleep(1)  # be polite to a shared public endpoint
    return aggregate(rows)


def main(argv: list[str] | None = None) -> int:
    """Harvest Wikidata for one language and write the stage-2 TSV.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True, choices=sorted(LANG_CODES))
    parser.add_argument("--page-size", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=BUILD_DIR)
    args = parser.parse_args(argv)

    rows = harvest(args.lang, page_size=args.page_size)
    path = args.out / f"wikidata.{args.lang}.tsv"
    count = write_rows(path, rows)
    print(f"wikidata/{args.lang}: {count:,} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
