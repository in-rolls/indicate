#!/usr/bin/env python
"""Held-out (out-of-sample) exact-match accuracy on the training corpus.

Reproduces the exact 80/20 split used by ``train.py`` (same seeds + max lengths)
and measures exact-match transliteration accuracy on the validation pairs the
model never trained on. Reports both single-word and full-entry accuracy.

This is a proxy for a clean gold benchmark; see ``training/eval.py`` for the
canonical Google Dakshina evaluation.

Example:
    python training/oos_eval.py --model hindi --sample 3000
    python training/oos_eval.py --model punjabi
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
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
from training.train import (  # noqa: E402
    TransliterationDataset,
    load_pairs,
    load_tokenizer,
)

DATA_DIR = REPO_ROOT / "indicate" / "data"
MODELS = {
    "hindi": {
        "lang": "hindi",
        "data": REPO_ROOT / "data" / "hindi.csv.gz",
        "input_vocab": DATA_DIR / "hindi_to_english" / "hindi_tokens.json",
        "target_vocab": DATA_DIR / "hindi_to_english" / "english_tokens.json",
        "max_input": 47,
        "max_output": 173,
    },
    "punjabi": {
        "lang": "punjabi",
        "data": REPO_ROOT / "data" / "punjabi.csv.gz",
        "input_vocab": DATA_DIR / "punjabi_to_english" / "punjabi_tokens.json",
        "target_vocab": DATA_DIR / "punjabi_to_english" / "english_tokens.json",
        "max_input": 32,
        "max_output": 32,
    },
}


def reproduce_val_pairs(cfg: dict, seed: int, val_frac: float) -> list[tuple[str, str]]:
    """Recreate the held-out validation pairs exactly as train.py splits them."""
    pairs = load_pairs(cfg["data"])
    random.seed(seed)
    random.shuffle(pairs)
    dataset = TransliterationDataset(
        pairs,
        load_tokenizer(str(cfg["input_vocab"])),
        load_tokenizer(str(cfg["target_vocab"])),
        cfg["max_input"],
        cfg["max_output"],
    )
    val_size = int(len(dataset) * val_frac)
    train_size = len(dataset) - val_size
    _, val_set = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return [dataset.kept_pairs[i] for i in val_set.indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="hindi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument(
        "--sample", type=int, default=3000, help="random subset of val pairs to score"
    )
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

    cfg = MODELS[args.model]
    beam = max(args.beam, 5) if args.rerank else args.beam
    reranker = None
    if args.rerank:
        targets = [eng for _, eng in load_pairs(cfg["data"])]
        reranker = Reranker(targets, alpha=args.alpha)

    val_pairs = reproduce_val_pairs(cfg, args.seed, args.val_frac)
    # Group to multi-reference: a source counts correct if it matches any reference.
    refs: dict[str, set[str]] = defaultdict(set)
    for source, english in val_pairs:
        refs[source].add(english)
    sources = sorted(refs)
    print(
        f"[{args.model}] held-out: {len(val_pairs)} pairs, {len(sources)} unique sources"
    )

    rng = random.Random(args.seed)
    scored = (
        sources
        if not args.sample or args.sample >= len(sources)
        else rng.sample(sources, args.sample)
    )

    dists: list[int] = []
    ref_lens: list[int] = []
    misses: list[tuple[str, str, str]] = []
    for source in tqdm(scored, desc="oos-eval"):
        # engine=("model",): this measures the model on held-out data; a hit
        # would return the training label and inflate the score.
        pred = indicate.transliterate(
            source, source=cfg["lang"], engine=("model",), beam=beam, reranker=reranker
        )
        dist, ref_len = score_word(pred, refs[source])
        dists.append(dist)
        ref_lens.append(ref_len)
        if dist != 0 and len(misses) < 25:
            misses.append((source, sorted(refs[source])[0], pred))

    stats = summarize(dists, ref_lens)
    print(f"\n{format_summary(stats, f'{args.model} held-out')}")
    print("(held-out silver labels; CER credits near-misses exact-match misses)")

    if args.plot:
        plot_distance_hist(
            dists, str(args.plot), f"{args.model} → English (held-out silver)"
        )
        print(f"Saved edit-distance histogram to {args.plot}")

    print("\nSample misses (source -> gold | pred):")
    for source, english, pred in misses[:15]:
        print(f"  {source}  ->  {english!r}  |  {pred!r}")


if __name__ == "__main__":
    main()
