"""Latency and coverage benchmark for the word-level lookup table.

Run::

    uv run --group train python training/bench_lookup.py --lang punjabi
    uv run --group train python training/bench_lookup.py --lang punjabi --json out.json

Committed rather than run ad hoc, so the numbers can be reproduced after any
change to the table, the normalizer or the splice. ``tests/test_lookup_bench.py``
asserts floors against the same functions so a regression fails CI.

Reports five things:

1. table size, load time and resident memory
2. lookup throughput under Zipfian access, which is how real corpora behave
3. hit rate over a slice of the electoral roll
4. end-to-end transliteration throughput with and without the table
5. whether an all-hit input can be served without importing torch at all
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indicate.lookup import Lookup, clear_cache, lookup_key  # noqa: E402

LANG_SUBDIR = {"punjabi": "punjabi_to_english", "hindi": "hindi_to_english"}
ROLL_PARQUET = REPO_ROOT / "data" / "punjab_transliteration_subset.parquet"
ROLL_COLUMNS = ["elector_name", "father_or_husband_name", "main_town", "district"]


def _rss_mb() -> float:
    """Return current resident set size in MB, or 0.0 if it cannot be read.

    Current, not peak: peak would charge the table for transient allocations
    during parsing, and what matters to a long-lived process is what the table
    still costs once it is loaded.
    """
    try:
        out = subprocess.run(  # noqa: S603
            ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(out.stdout.strip()) / 1024  # ps reports KB on macOS and Linux
    except Exception:  # pragma: no cover - platform dependent
        return 0.0


def measure_load(subdir: str) -> dict:
    """Time loading the table from disk.

    Args:
        subdir: Per-language data directory.

    Returns:
        Table size, load seconds and resident memory.
    """
    clear_cache()
    before = _rss_mb()
    start = time.perf_counter()
    table = Lookup.load(subdir)
    seconds = time.perf_counter() - start
    if table is None:
        return {"keys": 0, "load_s": 0.0, "rss_delta_mb": 0.0, "available": False}
    path = REPO_ROOT / "indicate" / "data" / subdir / "lookup.tsv.gz"
    return {
        "available": True,
        "keys": len(table),
        "convention": table.convention,
        "file_mb": round(path.stat().st_size / 1e6, 2) if path.is_file() else None,
        "load_s": round(seconds, 3),
        "rss_delta_mb": round(_rss_mb() - before, 1),
    }


def measure_throughput(subdir: str, draws: int = 300_000) -> dict:
    """Measure lookup throughput two ways, best case and worst case.

    ``zipf`` repeats a small hot vocabulary, which is how real corpora behave
    and which the key cache is built for. ``distinct`` never repeats a word, so
    every call pays full normalization; it is the floor, not the expectation.

    Args:
        subdir: Per-language data directory.
        draws: Number of lookups to perform in the Zipfian pass.

    Returns:
        Throughput in lookups per second for each access pattern.
    """
    table = Lookup.load(subdir)
    if table is None:
        return {"zipf_per_s": 0, "distinct_per_s": 0}
    keys = sorted(table._table)  # noqa: SLF001
    rng = random.Random(0)

    vocab = rng.sample(keys, min(5000, len(keys)))
    words = [
        vocab[min(int(rng.paretovariate(1.1)) - 1, len(vocab) - 1)] for _ in range(draws)
    ]
    start = time.perf_counter()
    for word in words:
        table.get(word)
    zipf = draws / (time.perf_counter() - start)

    # Each key is looked up once, and the cache is cleared first so no earlier
    # pass can have warmed it.
    lookup_key.cache_clear()
    start = time.perf_counter()
    for word in keys:
        table.get(word)
    distinct = len(keys) / (time.perf_counter() - start)

    return {"zipf_per_s": int(zipf), "distinct_per_s": int(distinct)}


def measure_coverage(subdir: str, rows: int = 200_000) -> dict:
    """Measure hit rate over a slice of the electoral roll.

    Args:
        subdir: Per-language data directory.
        rows: Roll rows to scan.

    Returns:
        Token counts and hit rate, or ``{}`` when the roll is unavailable.
    """
    table = Lookup.load(subdir)
    if table is None or not ROLL_PARQUET.is_file():
        return {}
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}

    hit = miss = seen = 0
    for batch in pq.ParquetFile(ROLL_PARQUET).iter_batches(
        batch_size=100_000, columns=ROLL_COLUMNS
    ):
        data = batch.to_pydict()
        for column in ROLL_COLUMNS:
            for value in data[column]:
                if not value:
                    continue
                for word in value.split():
                    if table.get(word) is not None:
                        hit += 1
                    else:
                        miss += 1
        seen += batch.num_rows
        if seen >= rows:
            break
    total = hit + miss
    return {
        "rows": seen,
        "tokens": total,
        "hit_pct": round(100 * hit / total, 2) if total else 0.0,
    }


def measure_end_to_end(lang: str, texts: list[str]) -> dict:
    """Transliterate the same texts with and without the table.

    Args:
        lang: Corpus language name.
        texts: Input strings.

    Returns:
        Throughput each way and the realized speedup.
    """
    import indicate

    tokens = sum(len(t.split()) for t in texts)

    start = time.perf_counter()
    indicate.transliterate_batch(texts, source=lang, engine=("lookup", "model"))
    with_lookup = time.perf_counter() - start

    start = time.perf_counter()
    indicate.transliterate_batch(texts, source=lang, engine=("model",))
    without = time.perf_counter() - start

    return {
        "tokens": tokens,
        "with_lookup_tok_s": int(tokens / with_lookup) if with_lookup else 0,
        "model_only_tok_s": int(tokens / without) if without else 0,
        "speedup": round(without / with_lookup, 1) if with_lookup else 0,
    }


def _cold_start_once(lang: str, text: str, *, engine: tuple[str, ...]) -> dict:
    """Run one fresh process and time import-through-first-answer.

    Args:
        lang: Corpus language name.
        text: An input every word of which is in the table.
        engine: The backend chain that process should use.

    Returns:
        Wall-clock seconds, whether torch was imported, and the output.
    """
    script = (
        "import sys, time\n"
        "t = time.perf_counter()\n"
        "import indicate\n"
        f"out = indicate.transliterate({text!r}, source={lang!r}, "
        f"engine={engine!r})\n"
        "print(time.perf_counter() - t, 'torch' in sys.modules, out)\n"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    if proc.returncode != 0:
        return {"error": (proc.stderr.strip().splitlines() or ["failed"])[-1]}
    seconds, torch_imported, *rest = proc.stdout.split()
    return {
        "seconds": round(float(seconds), 2),
        "torch_imported": torch_imported == "True",
        "output": " ".join(rest),
    }


def measure_cold_start(lang: str, text: str) -> dict:
    """Compare cold start with and without the table on an all-hit input.

    Both sides run as fresh subprocesses with a warm filesystem cache, so this
    is the steady-state cost a user pays per invocation, not a first-ever run.

    Args:
        lang: Corpus language name.
        text: An input every word of which is in the table.

    Returns:
        Both measurements and the speedup between them.
    """
    with_lookup = _cold_start_once(lang, text, engine=("lookup", "model"))
    model_only = _cold_start_once(lang, text, engine=("model",))
    out = {"with_lookup": with_lookup, "model_only": model_only}
    if "error" not in with_lookup and "error" not in model_only:
        out["speedup"] = round(model_only["seconds"] / with_lookup["seconds"], 1)
    return out


def roll_texts(limit: int = 20_000) -> list[str]:
    """Read name strings from the roll for the end-to-end benchmark.

    Args:
        limit: Maximum strings to return.

    Returns:
        Raw name strings, or ``[]`` when the roll is unavailable.
    """
    if not ROLL_PARQUET.is_file():
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return []
    out: list[str] = []
    for batch in pq.ParquetFile(ROLL_PARQUET).iter_batches(
        batch_size=50_000, columns=["elector_name"]
    ):
        out.extend(v for v in batch.to_pydict()["elector_name"] if v)
        if len(out) >= limit:
            break
    return out[:limit]


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print a report.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="punjabi", choices=sorted(LANG_SUBDIR))
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--texts", type=int, default=20_000)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    subdir = LANG_SUBDIR[args.lang]
    is_roll_lang = args.lang == "punjabi"
    report = {"lang": args.lang}
    report["table"] = measure_load(subdir)
    if not report["table"]["available"]:
        print(f"no lookup table for {args.lang}; build it first", file=sys.stderr)
        return 1

    report["throughput"] = measure_throughput(subdir)
    # The only large in-domain corpus to hand is the Punjab roll, which is
    # Gurmukhi; running it against the Hindi table would measure nothing.
    report["coverage"] = measure_coverage(subdir, args.rows) if is_roll_lang else {}
    texts = roll_texts(args.texts) if is_roll_lang else []
    if texts:
        report["end_to_end"] = measure_end_to_end(args.lang, texts)
    hit_word = "ਸਿੰਘ" if args.lang == "punjabi" else "गौरव"
    report["cold_start"] = measure_cold_start(args.lang, f"{hit_word} {hit_word}")

    table = report["table"]
    print(f"\n=== lookup benchmark: {args.lang} ===")
    print(
        f"  table        {table['keys']:,} keys, {table['file_mb']} MB, "
        f"load {table['load_s']}s, +{table['rss_delta_mb']} MB RSS, "
        f"convention={table['convention']}"
    )
    tp = report["throughput"]
    print(
        f"  throughput   {tp['zipf_per_s']:,} lookups/s (zipf, cached) / "
        f"{tp['distinct_per_s']:,} (every key distinct)"
    )
    if report["coverage"]:
        cov = report["coverage"]
        print(
            f"  coverage     {cov['hit_pct']}% of {cov['tokens']:,} roll tokens "
            f"over {cov['rows']:,} rows"
        )
    if report.get("end_to_end"):
        e2e = report["end_to_end"]
        print(
            f"  end-to-end   {e2e['with_lookup_tok_s']:,} tok/s with lookup vs "
            f"{e2e['model_only_tok_s']:,} model-only  ({e2e['speedup']}x)"
        )
    cold = report["cold_start"]
    if "speedup" in cold:
        hit, miss = cold["with_lookup"], cold["model_only"]
        print(
            f"  cold start   {hit['seconds']}s with lookup "
            f"(torch imported: {hit['torch_imported']}) vs "
            f"{miss['seconds']}s model-only  ({cold['speedup']}x)"
        )
        print(f"               -> {hit['output']!r} / {miss['output']!r}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
