"""Stage 3: assemble harvested claims into the published corpus, and report on it.

Run::

    uv run python -m gazetteer.build --lang hindi

Reads every redistributable source's stage-2 TSV, drops implausible pairs,
adjudicates the rest into ranked candidates with confidence tiers, joins the
stage-1 frequency evidence, and writes:

``gazetteer/build/corpus/<lang>.jsonl``
    The corpus, one JSON record per key.
``gazetteer/build/corpus/<lang>.report.json``
    The conviction report -- coverage by token mass at each tier, source
    agreement, and the contamination regression checks.

Only sources with ``redistributable=True`` are read, so the corpus can be
published under CC-BY-4.0. Frequency evidence may come from restricted corpora
because counts are aggregate; romanizations may not.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from gazetteer.adjudicate import HIGH, LOW, MEDIUM, Adjudication, adjudicate_all
from gazetteer.records import CandidateRow, read_rows
from gazetteer.script import is_language_script
from gazetteer.sources import bundled_sources
from indicate.normalize import NORMALIZER_VERSION

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "gazetteer" / "build" / "src"
FREQ_DIR = REPO_ROOT / "gazetteer" / "build" / "freq"
CORPUS_DIR = REPO_ROOT / "gazetteer" / "build" / "corpus"

#: Pairs an unscoped Wikidata pull asserted, which must never reach the corpus.
#: Each is a translation or a European homograph on a very frequent roll token.
#: Keyed by language, because a probe from another script would always pass and
#: a vacuous green check is worse than no check.
CONTAMINATION_PROBES: dict[str, tuple[tuple[str, str], ...]] = {
    "punjabi": (
        ("ਮਸੀਹ", "christ"),
        ("ਪਾਲ", "paul"),
        ("ਚੱਕ", "chuck"),
    ),
    "hindi": (
        ("मसीह", "christ"),
        ("पाल", "paul"),
    ),
}


def load_harvest(lang: str, src_dir: Path = SRC_DIR) -> list[CandidateRow]:
    """Read every redistributable source's rows for one language.

    Rows whose key is not written in this language's script are dropped. An
    unscoped Wikidata pull returns every script an entity is labelled in, and
    those cross-script rows are the ones that carry translations rather than
    romanizations.

    Args:
        lang: Corpus language name.
        src_dir: Directory holding stage-2 TSVs.

    Returns:
        All candidate rows from bundled sources, in this language's script.
    """
    rows: list[CandidateRow] = []
    for spec in bundled_sources():
        rows.extend(
            row
            for row in read_rows(src_dir / f"{spec.name}.{lang}.tsv")
            if is_language_script(row.native, lang)
        )
    return rows


def load_frequency(lang: str, freq_dir: Path = FREQ_DIR) -> dict[str, int]:
    """Read the stage-1 key counts.

    Args:
        lang: Corpus language name.
        freq_dir: Directory holding stage-1 TSVs.

    Returns:
        Key to occurrence count; empty if the file is absent.
    """
    path = freq_dir / f"{lang}.tsv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {
            r["key"]: int(r["count"]) for r in csv.DictReader(handle, delimiter="\t")
        }


def to_record(entry: Adjudication, lang: str, freq: int) -> dict:
    """Serialize one adjudicated key as a corpus record.

    Args:
        entry: The adjudicated key.
        lang: Corpus language name.
        freq: Corpus occurrence count for the key, or ``0`` if unmeasured.

    Returns:
        A JSON-serializable record.
    """
    return {
        "key": entry.key,
        "lang": lang,
        "normalizer": NORMALIZER_VERSION,
        "confidence": entry.confidence,
        "margin": round(entry.margin, 4),
        "freq": freq,
        "entity": dict(entry.entity),
        "candidates": [
            {
                "latin": c.latin,
                "score": round(c.score, 4),
                "sources": list(c.sources),
                "groups": list(c.groups),
            }
            for c in entry.candidates
        ],
    }


def build_report(
    entries: Sequence[Adjudication],
    frequency: Mapping[str, int],
    rows: Sequence[CandidateRow],
    lang: str,
) -> dict:
    """Measure whether the corpus is trustworthy enough to publish.

    Args:
        entries: Adjudicated keys.
        frequency: Stage-1 key counts.
        rows: The harvested rows the corpus was built from.
        lang: Corpus language name.

    Returns:
        The conviction report.
    """
    total_tokens = sum(frequency.values())
    by_tier: Counter[str] = Counter(e.confidence for e in entries)
    covered = {e.key for e in entries}

    def mass(keys) -> float:
        if not total_tokens:
            return 0.0
        return 100.0 * sum(frequency.get(k, 0) for k in keys) / total_tokens

    tier_keys = {
        tier: {e.key for e in entries if e.confidence == tier}
        for tier in (HIGH, MEDIUM, LOW)
    }
    contested = [e for e in entries if len(e.candidates) > 1]

    # Probes are language-scoped; applicable=0 means the check said nothing, not
    # that it passed. A probe counts as leaked only when the contaminant *wins*:
    # पाल really is both "pal" and "paul" (Indian Christians are named Paul), so
    # its mere presence is honest ambiguity, and the ranking is what matters.
    probes = CONTAMINATION_PROBES.get(lang, ())
    by_key = {e.key: e for e in entries}
    leaked, present = [], []
    for native, latin in probes:
        entry = by_key.get(native)
        if entry is None:
            continue
        if entry.candidates[0].latin == latin:
            leaked.append(f"{native}->{latin} ranked first")
        elif any(c.latin == latin for c in entry.candidates):
            present.append(f"{native}->{latin} present but outranked")
    probed = [n for n, _ in probes if n in covered]

    return {
        "lang": lang,
        "normalizer": NORMALIZER_VERSION,
        "sources": {
            spec.name: {
                "licence": spec.licence,
                "trust_group": spec.trust_group,
                "human_attested": spec.human_attested,
                "rows": sum(1 for r in rows if r.source == spec.name),
            }
            for spec in bundled_sources()
        },
        "frequency": {"keys": len(frequency), "tokens": total_tokens},
        "corpus": {
            "keys": len(entries),
            "by_confidence": dict(by_tier),
            "contested_keys": len(contested),
        },
        "coverage_pct_of_token_mass": {
            "any": round(mass(covered), 2),
            "high": round(mass(tier_keys[HIGH]), 2),
            "high_or_medium": round(mass(tier_keys[HIGH] | tier_keys[MEDIUM]), 2),
        },
        "contamination": {
            "probes_defined": len(probes),
            "probe_keys_present_in_corpus": len(probed),
            "leaked": leaked,
            "present_but_outranked": present,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Build the corpus and its conviction report for one language.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status; non-zero if a contamination probe leaked.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--src", type=Path, default=SRC_DIR)
    parser.add_argument("--freq", type=Path, default=FREQ_DIR)
    parser.add_argument("--out", type=Path, default=CORPUS_DIR)
    args = parser.parse_args(argv)

    rows = load_harvest(args.lang, args.src)
    if not rows:
        print(f"no harvested rows for {args.lang} in {args.src}")
        return 1
    frequency = load_frequency(args.lang, args.freq)
    entries = adjudicate_all(rows, lang=args.lang)

    args.out.mkdir(parents=True, exist_ok=True)
    corpus_path = args.out / f"{args.lang}.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            record = to_record(entry, args.lang, frequency.get(entry.key, 0))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = build_report(entries, frequency, rows, args.lang)
    (args.out / f"{args.lang}.report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"{args.lang}: {len(entries):,} keys -> {corpus_path}")
    for tier in (HIGH, MEDIUM, LOW):
        print(f"  {tier:7s} {report['corpus']['by_confidence'].get(tier, 0):>7,} keys")
    cov = report["coverage_pct_of_token_mass"]
    print(f"  token mass covered: any {cov['any']}%, high {cov['high']}%")
    contamination = report["contamination"]
    if contamination["leaked"]:
        print(f"  CONTAMINATION LEAKED: {contamination['leaked']}")
        return 2
    print(
        f"  contamination probes: {contamination['probe_keys_present_in_corpus']}"
        f"/{contamination['probes_defined']} applicable, none ranked first"
    )
    for note in contamination["present_but_outranked"]:
        print(f"    note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
