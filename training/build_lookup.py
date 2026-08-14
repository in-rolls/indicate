"""Build the word-level lookup table from a committed training corpus.

Run::

    uv run python training/build_lookup.py --lang punjabi
    uv run python training/build_lookup.py --lang hindi
    uv run python training/build_lookup.py --lang punjabi --eval-clean

Writes ``indicate/data/<lang>_to_english/lookup.tsv.gz``.

Source choice, measured on 1M rows of the Punjab roll:

===========================  =======  ====================
corpus                       keys     roll token mass hit
===========================  =======  ====================
data/punjabi.csv.gz          287,092  99.40%
data/pa_train_v2.csv.gz      693,732  78.37%
union                        698,209  99.49%
===========================  =======  ====================

The committed corpus wins by 21 points despite being smaller, because
``build_v2.py`` withholds every source word that appears in the evaluation sets
-- which removed the most frequent tokens in the corpus (``ਕੌਰ``, ``ਕੁਮਾਰ``,
``ਦਾਸ``, ``ਬਟਾਲਾ``, ``ਗੁਰਦਾਸਪੁਰ``). So we build from the committed corpora and
ignore the v2 files, whose extra 0.09% is not worth the size.

The shipped table reaches 99.12% rather than 99.40% because :func:`choose`
declines to answer where the corpus has no majority. That trade is measured and
deliberate: see its docstring.

``--eval-clean`` excludes every source word in the blended eval set and the
Dakshina test split, for benchmarking. Scoring the ordinary table against those
sets would measure memorization.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gazetteer.plausibility import aksharas  # noqa: E402
from indicate.lookup import FORMAT_VERSION, LOOKUP_FILE, lookup_key  # noqa: E402
from indicate.normalize import NORMALIZER_VERSION, latin_form  # noqa: E402

#: Most Latin characters one akshara can justify. The corpus 99.9th percentile
#: is 4.0; above it are alignment failures, not spellings (``हैं`` -> ``ghana``).
MAX_RATIO = 4.0

#: Per-language source corpus, its columns, and the convention it speaks.
CORPORA = {
    "punjabi": {
        "path": "data/punjabi.csv.gz",
        "native": "punjabi",
        "latin": "english",
        "convention": "roll",
        "subdir": "punjabi_to_english",
    },
    "hindi": {
        "path": "data/hindi.csv.gz",
        "native": "hindi",
        "latin": "english",
        # Not "roll": this corpus blends Cricinfo, election affidavits, Dakshina
        # and IIT Bombay, so it has no single romanization house style.
        "convention": "mixed",
        "subdir": "hindi_to_english",
    },
}

#: Files whose source words must be excluded from an --eval-clean table.
EVAL_SOURCES = {
    "punjabi": (
        "data/eval/pa_blended.tsv",
        "data/dakshina/pa.translit.sampled.test.tsv",
    ),
    "hindi": ("data/eval/hi_blended.tsv", "data/dakshina/hi.translit.sampled.test.tsv"),
}


def read_eval_keys(lang: str, root: Path) -> set[str]:
    """Collect the normalized source words used by the evaluation sets.

    Args:
        lang: Corpus language name.
        root: Repository root.

    Returns:
        Normalized keys to withhold; empty when no eval file is present.
    """
    keys: set[str] = set()
    for rel in EVAL_SOURCES.get(lang, ()):
        path = root / rel
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                first = line.split("\t", 1)[0].strip()
                for word in first.split():
                    key = lookup_key(word)
                    if key:
                        keys.add(key)
    return keys


def is_implausible(key: str, value: str) -> bool:
    """Report whether ``value`` is too short or too long to be ``key`` romanized.

    Length is the only signal available at build time that does not consult the
    model, so it is the only one used. The band is loose on purpose: it exists
    to rule out alignment debris, not to adjudicate spelling.

    A one-character romanization of a multi-akshara word is the debris signature
    (``ਰਾਜ`` -> ``j``). It is not universally wrong -- ``आर`` is how Devanagari
    writes the letter *R* -- which is why this only ever breaks a tie and never
    overrides a vote that had a winner.

    Args:
        key: Normalized native-script key.
        value: Candidate romanization.

    Returns:
        True when the pair cannot plausibly be a transliteration.
    """
    letters = aksharas(key)
    if not letters:
        return True
    if len(value) == 1 and letters >= 2:
        return True
    return len(value) / letters > MAX_RATIO


def choose(key: str, counter: Counter[str]) -> str | None:
    """Pick one romanization for a key, or ``None`` if the corpus has no opinion.

    A modal vote with an outright winner is taken. A **tie is not broken by
    fiat**: returning ``None`` drops the key from the table so the word falls
    through to the model, which is measurably the better answer. On the 1,154
    Hindi words shared by the table and the Dakshina test set:

    ============================================  ======
    rule                                          exact
    ============================================  ======
    table always answers (alphabetical tie-break)  76.6%
    model only                                     76.2%
    table only when it has a majority, else model  78.8%
    ============================================  ======

    21% of those Hindi words are ties, because the Hindi corpus is phrase-level
    and positional alignment yields one attestation per pair: ``मुंबई`` has
    ``mumbai``, ``mumabi``, ``mumba`` and ``mumbaikar`` all at one vote each, and
    alphabetical order picks the typo. Punjabi is word-level, so only 0.04% of
    its keys tie and the rule costs it nothing.

    The one exception is a tie in which plausibility eliminates all but one
    candidate. That is not a real disagreement, it is a real answer next to
    alignment debris (``ਰਾਜ``: ``raaj`` against ``j``), so it stands.

    Sorting rather than relying on ``most_common`` insertion order is what makes
    a rebuild byte-identical.

    Args:
        key: Normalized native-script key.
        counter: Candidate romanizations and their attestation counts.

    Returns:
        The winning romanization, or ``None`` to leave the key out.
    """
    top = max(counter.values())
    tied = sorted(value for value, count in counter.items() if count == top)
    if len(tied) == 1:
        return tied[0]
    plausible = [value for value in tied if not is_implausible(key, value)]
    return plausible[0] if len(plausible) == 1 else None


def build_table(path: Path, native: str, latin: str) -> tuple[dict[str, str], int, int]:
    """Build a key to romanization table by modal vote over aligned word pairs.

    Rows whose two sides have different token counts are skipped: positional
    correspondence is only meaningful when the counts match, and a guess there
    would poison a table that is otherwise authoritative.

    Args:
        path: Gzipped CSV corpus.
        native: Native-script column name.
        latin: Latin column name.

    Returns:
        The table, the number of keys that had more than one candidate, and the
        number left out because the vote tied.
    """
    votes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            native_tokens = (row.get(native) or "").split()
            latin_tokens = (row.get(latin) or "").split()
            if not native_tokens or len(native_tokens) != len(latin_tokens):
                continue
            for native_token, latin_token in zip(
                native_tokens, latin_tokens, strict=True
            ):
                key = lookup_key(native_token)
                value = latin_form(latin_token)
                if key and value:
                    votes[key][value] += 1

    contested = sum(1 for counter in votes.values() if len(counter) > 1)
    table = {
        key: value
        for key, counter in votes.items()
        if (value := choose(key, counter)) is not None
    }
    return table, contested, len(votes) - len(table)


def write_table(path: Path, table: dict[str, str], meta: dict[str, str]) -> int:
    """Write the table as a gzipped TSV with a versioned header.

    Args:
        path: Destination ``lookup.tsv.gz``.
        table: Key to romanization.
        meta: Header fields.

    Returns:
        Bytes written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"#{field}\t{value}" for field, value in meta.items()]
    lines += [f"{key}\t{table[key]}" for key in sorted(table)]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    # mtime=0 so a rebuild from the same corpus is byte-identical.
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    return path.stat().st_size


def main(argv: list[str] | None = None) -> int:
    """Build and write one language's lookup table.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True, choices=sorted(CORPORA))
    parser.add_argument(
        "--eval-clean",
        action="store_true",
        help="exclude eval-set source words, for honest benchmarking",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    spec = CORPORA[args.lang]
    corpus = REPO_ROOT / spec["path"]
    if not corpus.is_file():
        print(f"missing corpus: {corpus}", file=sys.stderr)
        return 1

    table, contested, undecided = build_table(corpus, spec["native"], spec["latin"])
    withheld = 0
    if args.eval_clean:
        eval_keys = read_eval_keys(args.lang, REPO_ROOT)
        before = len(table)
        table = {k: v for k, v in table.items() if k not in eval_keys}
        withheld = before - len(table)

    name = LOOKUP_FILE if not args.eval_clean else "lookup.eval-clean.tsv.gz"
    out = args.out or (REPO_ROOT / "indicate" / "data" / spec["subdir"] / name)
    meta = {
        "indicate-lookup": str(FORMAT_VERSION),
        "normalizer": str(NORMALIZER_VERSION),
        "lang": args.lang,
        "convention": spec["convention"],
        "source": spec["path"],
        "entries": str(len(table)),
        "eval_clean": "1" if args.eval_clean else "0",
    }
    size = write_table(out, table, meta)

    print(f"{args.lang}: {len(table):,} keys -> {out}")
    print(f"  contested keys (>1 candidate, modal vote applied): {contested:,}")
    print(f"  keys left to the model (vote tied, no majority): {undecided:,}")
    if args.eval_clean:
        print(f"  withheld as eval sources: {withheld:,}")
    print(f"  size: {size / 1e6:.2f} MB  convention: {spec['convention']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
