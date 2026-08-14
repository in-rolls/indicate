"""Stage 4: is the corpus actually right? Score it against a neutral reference.

Run::

    uv run python -m gazetteer.conviction --lang hindi

``gazetteer/build.py`` reports coverage, source agreement and contamination.
Those say the corpus is *internally* coherent. None of them says it is correct.
This does, by scoring its top-ranked romanization against the Google Dakshina
romanization lexicon -- a reference that is **not one of its sources**, which the
run asserts rather than assumes before reporting a single number.

Three things are measured, because they answer different questions:

**Accuracy by confidence tier.** If the tiers mean anything, ``high`` must beat
``medium``. A tier that does not separate is a tier worth deleting.

**Accuracy against the shipped model on the same words.** This is the decision.
A gazetteer that is no better than the model already in the wheel is not worth
publishing, however clean its provenance.

**Contested keys specifically.** Where sources disagree, adjudication is doing
real work; everywhere else it is copying. Scoring the contested subset separately
shows whether the ranking earns its complexity.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from gazetteer.sources import bundled_sources
from indicate.lookup import lookup_key
from indicate.normalize import latin_form

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "gazetteer" / "build" / "corpus"
DAKSHINA_DIR = REPO_ROOT / "data" / "dakshina"

#: Dakshina's language code for each corpus language.
LANGS = {"hindi": "hi", "punjabi": "pa"}


def reference_is_not_a_source(lang: str) -> None:
    """Fail loudly if Dakshina is among the corpus's own sources.

    Scoring a corpus against something it was built from measures memory, not
    accuracy. This is asserted at run time rather than trusted, because a source
    can be added later by someone who never reads this module.

    Args:
        lang: Corpus language name.

    Raises:
        AssertionError: If any bundled source looks Dakshina-derived.
    """
    names = {spec.name.lower() for spec in bundled_sources()}
    overlap = {name for name in names if "dakshina" in name}
    if overlap:
        raise AssertionError(
            f"cannot score {lang} against Dakshina: it is a corpus source ({overlap})"
        )


def load_gold(code: str) -> dict[str, set[str]]:
    """Read the Dakshina test split as key to accepted romanizations.

    Args:
        code: Dakshina language code, e.g. ``"hi"``.

    Returns:
        Normalized key to the set of accepted romanizations; ``{}`` if absent.
    """
    path = DAKSHINA_DIR / f"{code}.translit.sampled.test.tsv"
    if not path.is_file():
        return {}
    gold: defaultdict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            key = lookup_key(parts[0])
            value = latin_form(parts[1])
            if key and value:
                gold[key].add(value)
    return dict(gold)


def load_corpus(lang: str, corpus_dir: Path = CORPUS_DIR) -> list[dict]:
    """Read the built corpus.

    Args:
        lang: Corpus language name.
        corpus_dir: Directory holding ``<lang>.jsonl``.

    Returns:
        One record per key; ``[]`` if the corpus has not been built.
    """
    path = corpus_dir / f"{lang}.jsonl"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rate(hits: int, total: int) -> float:
    return round(100.0 * hits / total, 1) if total else 0.0


def score(lang: str, records: list[dict], gold: dict[str, set[str]]) -> dict:
    """Score the corpus, and the shipped model, on the words they share.

    Args:
        lang: Corpus language name.
        records: Corpus records.
        gold: Reference romanizations by key.

    Returns:
        Overlap size and accuracy overall, by tier, and for contested keys.
    """
    overlap = [r for r in records if r["key"] in gold and r["candidates"]]
    if not overlap:
        return {"overlap": 0}

    import indicate

    keys = [r["key"] for r in overlap]
    # engine=("model",): the packaged table shares corpora with these sources,
    # so leaving it in would compare the gazetteer against a second lookup.
    predictions = indicate.transliterate_batch(keys, source=lang, engine=("model",))

    corpus_hit = model_hit = 0
    tiers: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    contested = [0, 0]
    for record, prediction in zip(overlap, predictions, strict=True):
        accepted = gold[record["key"]]
        right = record["candidates"][0]["latin"] in accepted
        corpus_hit += right
        model_hit += latin_form(str(prediction)) in accepted
        tier = tiers[record["confidence"]]
        tier[0] += right
        tier[1] += 1
        if len(record["candidates"]) > 1:
            contested[0] += right
            contested[1] += 1

    return {
        "overlap": len(overlap),
        "corpus_pct": _rate(corpus_hit, len(overlap)),
        "model_pct": _rate(model_hit, len(overlap)),
        "by_confidence": {
            tier: {"keys": count, "pct": _rate(hit, count)}
            for tier, (hit, count) in sorted(tiers.items())
        },
        "contested": {"keys": contested[1], "pct": _rate(contested[0], contested[1])},
    }


def main(argv: list[str] | None = None) -> int:
    """Score one language's corpus and print the report.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True, choices=sorted(LANGS))
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    args = parser.parse_args(argv)

    reference_is_not_a_source(args.lang)
    records = load_corpus(args.lang, args.corpus)
    if not records:
        print(f"no corpus for {args.lang}; run gazetteer.build first")
        return 1
    gold = load_gold(LANGS[args.lang])
    if not gold:
        print(f"no Dakshina test split for {args.lang}; see training/README.md")
        return 1

    result = score(args.lang, records, gold)
    if not result["overlap"]:
        print(f"{args.lang}: corpus and Dakshina test set share no keys")
        return 1

    print(f"\n=== conviction: {args.lang} ===")
    print(f"  {result['overlap']:,} keys in both the corpus and Dakshina test")
    print(f"  corpus top-1   {result['corpus_pct']}%")
    print(f"  shipped model  {result['model_pct']}%  (same words, lookup off)")
    print("\n  by confidence tier:")
    for tier, stats in result["by_confidence"].items():
        print(f"    {tier:8s} {stats['keys']:6,} keys   {stats['pct']}%")
    if result["contested"]["keys"]:
        print(
            f"\n  contested keys (>1 candidate): "
            f"{result['contested']['keys']:,} keys   {result['contested']['pct']}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
