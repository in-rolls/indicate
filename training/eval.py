#!/usr/bin/env python
"""Evaluate exact-match transliteration accuracy on the Dakshina test set.

The Google Dakshina romanization lexicon is not committed to this repo (see
``training/README.md`` for the one-time download). Point ``--test-file`` at
``hi.translit.sampled.test.tsv`` whose tab-separated columns are
``<native (hindi)>\t<romanization (english)>\t<count>``.

A native word counts as correct if the model's output matches any reference
romanization for that word.

Example:
    python training/eval.py --test-file path/to/hi.translit.sampled.test.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import indicate  # noqa: E402
from indicate.rerank import Reranker  # noqa: E402
from training.metrics import (  # noqa: E402
    format_summary,
    plot_distance_hist,
    score_word,
    summarize,
)
from training.train import load_pairs  # noqa: E402

DAKSHINA_DIR = REPO_ROOT / "data" / "dakshina"
DATA_DIR = REPO_ROOT / "data"
MODELS = {
    "hindi": (
        "hindi",
        DAKSHINA_DIR / "hi.translit.sampled.test.tsv",
        DATA_DIR / "hindi.csv.gz",
    ),
    "punjabi": (
        "punjabi",
        DAKSHINA_DIR / "pa.translit.sampled.test.tsv",
        DATA_DIR / "punjabi.csv.gz",
    ),
}


def load_references(path: Path) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            hindi, english = row[0].strip(), row[1].strip()
            if hindi and english:
                refs[hindi].add(english)
    return refs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="hindi")
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="defaults to the Dakshina test set for --model",
    )
    parser.add_argument("--limit", type=int, default=0, help="cap #words (smoke test)")
    parser.add_argument("--beam", type=int, default=1, help="beam width (1 = greedy)")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="re-rank beam top-k with a char-LM built from training targets",
    )
    parser.add_argument("--alpha", type=float, default=0.9, help="rerank interpolation")
    parser.add_argument(
        "--plot", type=Path, default=None, help="save the edit-distance histogram PNG"
    )
    args = parser.parse_args()

    lang, default_test, train_csv = MODELS[args.model]
    beam = max(args.beam, 5) if args.rerank else args.beam
    reranker = None
    if args.rerank:
        targets = [eng for _, eng in load_pairs(train_csv)]
        reranker = Reranker(targets, alpha=args.alpha)
    test_file = args.test_file or default_test

    refs = load_references(test_file)
    words = list(refs)
    if args.limit:
        words = words[: args.limit]
    print(
        f"[{args.model}] evaluating {len(words)} unique words from {test_file} "
        f"(beam={beam}, rerank={args.rerank})"
    )

    dists: list[int] = []
    ref_lens: list[int] = []
    for word in tqdm(words, desc="eval"):
        # engine=("model",): the table is built from the training corpus, so
        # leaving it in would score memorization rather than the model.
        pred = indicate.transliterate(
            word, source=lang, engine=("model",), beam=beam, reranker=reranker
        )
        dist, ref_len = score_word(pred, refs[word])
        dists.append(dist)
        ref_lens.append(ref_len)

    stats = summarize(dists, ref_lens)
    print(format_summary(stats, args.model))

    if args.plot:
        plot_distance_hist(
            dists, str(args.plot), f"{args.model} → English (Dakshina test)"
        )
        print(f"Saved edit-distance histogram to {args.plot}")


if __name__ == "__main__":
    main()
