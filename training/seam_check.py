"""Measure the mixing seam: does splicing table hits into model output show?

Run::

    uv run --group train python training/seam_check.py --lang punjabi

The lookup table and the model romanize in different house styles. The table
inherits the long-vowel style of its corpus annotation (``azaadi``, ``apaar``,
``ambee``); the model tends shorter (``azadi``, ``apar``, ``ambi``). Aggregate
accuracy cannot see this, because both are defensible spellings of the same
word. What it can produce is a string whose halves disagree — a table word
next to a model word, in visibly different styles.

The measurement has a control, which is what makes it falsifiable. For each
input we render it twice: **spliced** (table for hits, model for misses) and
**model-only** (everything decoded). Then we compare the hit words against the
miss words *within each rendering*.

In the model-only rendering both halves come from the same source, so any style
gap between them is a property of which words happen to hit, not of splicing.
Subtracting that baseline gap from the spliced gap isolates the seam. If the
excess is ~0, splicing is invisible and the on-by-default choice is safe. If it
is large, it is not.

Two style statistics, deliberately different in what they assume:

- **long-vowel rate**: doubled vowel digraphs per character. Directly tests the
  stated hypothesis, but presumes the difference is about vowel length.
- **letters per akshara**: output characters divided by input characters.
  Assumes nothing about which letters; a systematically wordier romanization
  shows up here whatever its cause.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import indicate  # noqa: E402
from indicate.lookup import Lookup  # noqa: E402

ROLL_PARQUET = REPO_ROOT / "data" / "punjab_transliteration_subset.parquet"
ROLL_COLUMNS = ["elector_name", "father_or_husband_name", "main_town", "district"]

LONG_VOWEL = re.compile(r"aa|ee|ii|oo|uu")

LANGS = ("punjabi", "hindi")


def long_vowel_rate(word: str) -> float:
    """Doubled vowel digraphs per character.

    Args:
        word: A romanized word.

    Returns:
        Rate in [0, 1], or 0.0 for an empty word.
    """
    return len(LONG_VOWEL.findall(word)) / len(word) if word else 0.0


def letters_per_akshara(native: str, latin: str) -> float:
    """Romanized length divided by source length.

    Args:
        native: The source word.
        latin: Its romanization.

    Returns:
        Ratio, or 0.0 when the source is empty.
    """
    return len(latin) / len(native) if native else 0.0


def mixed_strings(table: Lookup, limit: int, seed: int = 0) -> list[str]:
    """Sample roll strings that contain both a table hit and a table miss.

    Args:
        table: The lookup table deciding hits.
        limit: How many strings to return.
        seed: Sampling seed.

    Returns:
        Raw roll strings, or ``[]`` when the roll is unavailable.
    """
    if not ROLL_PARQUET.is_file():
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return []

    rng = random.Random(seed)
    found: list[str] = []
    for batch in pq.ParquetFile(ROLL_PARQUET).iter_batches(
        batch_size=50_000, columns=ROLL_COLUMNS
    ):
        data = batch.to_pydict()
        for column in ROLL_COLUMNS:
            for value in data[column]:
                if not value:
                    continue
                words = value.split()
                if len(words) < 2:
                    continue
                marks = [table.get(w) is not None for w in words]
                if any(marks) and not all(marks):
                    found.append(value)
        if len(found) >= limit * 20:
            break
    rng.shuffle(found)
    return found[:limit]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compare(lang: str, texts: list[str]) -> dict:
    """Render each text both ways and measure the style gap in each.

    Args:
        lang: Corpus language name.
        texts: Roll strings containing both hits and misses.

    Returns:
        Gap statistics, agreement on hit words, and example renderings.
    """
    from indicate.languages import PAIRS

    table = Lookup.load(PAIRS[(lang, "english")].subdir)
    assert table is not None  # noqa: S101 - caller checked

    spliced = indicate.transliterate_batch(
        texts, source=lang, engine=("lookup", "model")
    )
    model_only = indicate.transliterate_batch(texts, source=lang, engine=("model",))

    stats: dict[str, list[float]] = {
        "spliced_hit_lv": [],
        "spliced_miss_lv": [],
        "model_hit_lv": [],
        "model_miss_lv": [],
        "spliced_hit_lpa": [],
        "spliced_miss_lpa": [],
        "model_hit_lpa": [],
        "model_miss_lpa": [],
    }
    agree = differ = 0
    disagreements: list[tuple[str, str, str]] = []

    for text, out_spliced, out_model in zip(texts, spliced, model_only, strict=True):
        words = text.split()
        sp = str(out_spliced).split()
        mo = str(out_model).split()
        if len(sp) != len(words) or len(mo) != len(words):
            continue  # a dropped word would misalign the comparison
        for native, sp_word, mo_word in zip(words, sp, mo, strict=True):
            side = "hit" if table.get(native) is not None else "miss"
            stats[f"spliced_{side}_lv"].append(long_vowel_rate(sp_word))
            stats[f"model_{side}_lv"].append(long_vowel_rate(mo_word))
            stats[f"spliced_{side}_lpa"].append(letters_per_akshara(native, sp_word))
            stats[f"model_{side}_lpa"].append(letters_per_akshara(native, mo_word))
            if side == "hit":
                if sp_word == mo_word:
                    agree += 1
                else:
                    differ += 1
                    if len(disagreements) < 400:
                        disagreements.append((native, sp_word, mo_word))

    means = {key: _mean(values) for key, values in stats.items()}
    hits = agree + differ
    return {
        "texts": len(texts),
        "hit_words": hits,
        "table_model_agree_pct": round(100 * agree / hits, 1) if hits else 0.0,
        "gap": {
            "long_vowel": {
                "spliced": round(means["spliced_hit_lv"] - means["spliced_miss_lv"], 4),
                "model_only": round(means["model_hit_lv"] - means["model_miss_lv"], 4),
            },
            "letters_per_akshara": {
                "spliced": round(
                    means["spliced_hit_lpa"] - means["spliced_miss_lpa"], 4
                ),
                "model_only": round(means["model_hit_lpa"] - means["model_miss_lpa"], 4),
            },
        },
        "examples": list(zip(texts[:12], spliced[:12], model_only[:12], strict=False)),
        "disagreements": disagreements[:20],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the seam check and print a report.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="punjabi", choices=sorted(LANGS))
    parser.add_argument("--texts", type=int, default=2000)
    args = parser.parse_args(argv)

    from indicate.languages import PAIRS

    table = Lookup.load(PAIRS[(args.lang, "english")].subdir)
    if table is None:
        print(f"no lookup table for {args.lang}", file=sys.stderr)
        return 1
    texts = mixed_strings(table, args.texts)
    if not texts:
        print("no mixed roll strings available", file=sys.stderr)
        return 1

    report = compare(args.lang, texts)

    print(f"\n=== mixing seam: {args.lang} ===")
    print(f"  {report['texts']:,} strings with both a hit and a miss")
    print(
        f"  on the {report['hit_words']:,} hit words, table and model agree "
        f"exactly {report['table_model_agree_pct']}% of the time"
    )
    print("\n  style gap between hit words and miss words (hit minus miss):")
    for name, pair in report["gap"].items():
        excess = pair["spliced"] - pair["model_only"]
        print(
            f"    {name:<20} spliced {pair['spliced']:+.4f}   "
            f"model-only {pair['model_only']:+.4f}   excess {excess:+.4f}"
        )
    print("\n  hit words where the two sources disagree:")
    for native, sp_word, mo_word in report["disagreements"]:
        print(f"    {native}  table={sp_word:<16} model={mo_word}")
    print("\n  examples (input / spliced / model-only):")
    for text, sp, mo in report["examples"]:
        print(f"    {text}\n      spliced : {sp}\n      model   : {mo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
