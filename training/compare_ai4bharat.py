#!/usr/bin/env python
"""Bake-off: AI4Bharat IndicXlit vs the shipped ``indicate`` model on a gold set.

IndicXlit (``ai4bharat-transliteration``) depends on ``fairseq``, which does not
import on Python 3.13. So this runs in two stages, each in its own interpreter:

1. Under a Python 3.10 venv with ``ai4bharat-transliteration`` installed::

       /tmp/xlit-venv/bin/python training/compare_ai4bharat.py gen-indicxlit \\
           --eval data/dakshina/pa.translit.sampled.test.tsv \\
           --out data/eval/pa_indicxlit_preds.tsv --lang pa --beam 4

2. Under the project's venv (Python 3.13)::

       python training/compare_ai4bharat.py compare \\
           --eval data/dakshina/pa.translit.sampled.test.tsv \\
           --indicxlit-preds data/eval/pa_indicxlit_preds.tsv --model punjabi --beam 5

Gold formats supported: Dakshina ``native<TAB>roman<TAB>count`` (multi-row per native)
and blended ``native<TAB>ref1<TAB>ref2...``. Scoring is top-1 exact-match + Acc@<=1 +
CER (multi-reference), via ``training/metrics.py``.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from collections import defaultdict
from pathlib import Path

# First-column values that indicate a header row to skip (corpus CSVs).
_HEADER_FIRST = {
    "punjabi",
    "assamese",
    "hindi",
    "bengali",
    "gurmukhi",
    "telugu",
    "native",
    "source",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_refs(path: Path) -> dict[str, set[str]]:
    """native -> set of accepted (lowercased) romanizations.

    Handles Dakshina ``native<TAB>roman<TAB>count`` (count dropped), blended
    ``native<TAB>ref1<TAB>ref2...``, and the gzipped corpus CSV
    ``native,english`` (comma-separated, with a header row to skip).
    """
    name = path.name
    delimiter = "," if ".csv" in name else "\t"
    opener = gzip.open if name.endswith(".gz") else open
    refs: dict[str, set[str]] = defaultdict(set)
    with opener(path, "rt", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            parts = [p.strip() for p in line.rstrip("\n").split(delimiter)]
            if i == 0 and parts and parts[0].lower() in _HEADER_FIRST:
                continue  # corpus header row
            if len(parts) < 2 or not parts[0]:
                continue
            native = parts[0]
            for token in parts[1:]:
                if token and not token.isdigit():  # skip Dakshina's count column
                    refs[native].add(token.lower())
    return refs


def _first_pred(out: object) -> str:
    """Normalise IndicXlit's return (list / dict[lang]->list / str) to a top-1 string."""
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        for value in out.values():
            return _first_pred(value)
        return ""
    if isinstance(out, (list, tuple)):
        return _first_pred(out[0]) if out else ""
    return str(out)


def cmd_gen_indicxlit(args: argparse.Namespace) -> None:
    # IndicXlit's dependency stack is fragile on modern envs. Two shims (safe here --
    # we only do Punjabi, and the AI4Bharat checkpoint is trusted):
    import sys as _sys
    import types as _types

    import torch as _torch

    # 1. Skip the Urdu-only `urduhack` import (drags tensorflow/keras).
    _sys.modules.setdefault("urduhack", _types.ModuleType("urduhack"))
    _sys.modules["urduhack"].normalize = lambda x: x  # type: ignore[attr-defined]
    # 2. torch>=2.6 defaults torch.load(weights_only=True), which rejects fairseq ckpts.
    _orig_load = _torch.load
    _torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

    from ai4bharat.transliteration import XlitEngine

    refs = load_refs(args.eval)
    natives = sorted(refs)
    if args.sample and args.sample < len(natives):
        import random

        natives = random.Random(42).sample(natives, args.sample)

    engine = XlitEngine(beam_width=args.beam, src_script_type="indic")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for i, native in enumerate(natives, 1):
            pred = _first_pred(
                engine.translit_word(native, lang_code=args.lang, topk=1)
            )
            handle.write(f"{native}\t{pred.lower()}\n")
            if i % 200 == 0:
                print(f"  {i}/{len(natives)}", file=sys.stderr)
    print(f"Wrote {len(natives)} IndicXlit predictions -> {args.out}")


def _load_preds(path: Path | None) -> dict[str, str]:
    preds: dict[str, str] = {}
    if path and path.is_file():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    preds[parts[0]] = parts[1].strip().lower()
    return preds


def cmd_gen_llm(args: argparse.Namespace) -> None:
    from indicate.llm_indic import IndicLLMTransliterator

    refs = load_refs(args.eval)
    natives = sorted(refs)
    if args.sample and args.sample < len(natives):
        import random

        natives = random.Random(args.seed).sample(natives, args.sample)

    trans = IndicLLMTransliterator(
        args.lang, "english", provider="openai", model=args.model
    )
    preds = trans.transliterate_batch(
        natives, batch_size=args.batch_size, use_few_shot=True
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for native, pred in zip(natives, preds, strict=False):
            handle.write(f"{native}\t{(pred or '').strip().lower()}\n")
    print(f"Wrote {len(natives)} LLM ({args.model}) predictions -> {args.out}")


def cmd_compare(args: argparse.Namespace) -> None:
    from training.metrics import format_summary, score_word, summarize

    refs = load_refs(args.eval)
    indicxlit_preds = _load_preds(args.indicxlit_preds)
    llm_preds = _load_preds(getattr(args, "llm_preds", None))

    # Score a common set: prefer the smallest provided contestant (e.g. an LLM run
    # over a 100-word sample) so every model is judged on identical words.
    if llm_preds:
        natives = sorted(llm_preds)
    elif indicxlit_preds:
        natives = sorted(indicxlit_preds)
    else:
        natives = sorted(refs)
    if args.sample and args.sample < len(natives):
        import random

        natives = random.Random(42).sample(natives, args.sample)
    print(f"Scoring {len(natives)} unique native words from {args.eval.name}")

    contestants: dict[str, dict[str, str]] = {}

    if llm_preds:
        contestants[f"llm:{args.llm_label}"] = {
            n: llm_preds.get(n, "") for n in natives
        }
    if indicxlit_preds:
        contestants["ai4bharat-indicxlit"] = {
            n: indicxlit_preds.get(n, "") for n in natives
        }

    if args.model and args.model != "none":
        import indicate
        from tqdm import tqdm

        preds = {}
        for native in tqdm(natives, desc=f"indicate:{args.model}"):
            # engine=("model",) so indicate competes as a model, not a table.
            preds[native] = str(
                indicate.transliterate(
                    native, source=args.model, engine=("model",), beam=args.beam
                )
            ).lower()
        contestants[f"indicate:{args.model}(beam={args.beam})"] = preds

    print()
    for label, preds in contestants.items():
        dists, ref_lens = [], []
        for native in natives:
            dist, ref_len = score_word(preds[native], refs[native])
            dists.append(dist)
            ref_lens.append(ref_len)
        print(format_summary(summarize(dists, ref_lens), label))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser(
        "gen-indicxlit", help="run IndicXlit -> predictions TSV (py3.10 venv)"
    )
    g.add_argument("--eval", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--lang", default="pa")
    g.add_argument("--beam", type=int, default=4)
    g.add_argument("--sample", type=int, default=None)
    g.set_defaults(func=cmd_gen_indicxlit)

    m = sub.add_parser("gen-llm", help="run an LLM (e.g. gpt-4o) -> predictions TSV")
    m.add_argument("--eval", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--lang", default="bengali")
    m.add_argument("--model", default="gpt-4o")
    m.add_argument("--batch-size", type=int, default=20)
    m.add_argument("--sample", type=int, default=100)
    m.add_argument("--seed", type=int, default=42)
    m.set_defaults(func=cmd_gen_llm)

    c = sub.add_parser(
        "compare", help="score indicate + IndicXlit + LLM preds against gold"
    )
    c.add_argument("--eval", type=Path, required=True)
    c.add_argument("--indicxlit-preds", type=Path, default=None)
    c.add_argument("--llm-preds", type=Path, default=None)
    c.add_argument("--llm-label", default="gpt-4o")
    c.add_argument("--model", choices=["punjabi", "hindi", "none"], default="punjabi")
    c.add_argument("--beam", type=int, default=5)
    c.add_argument("--sample", type=int, default=None)
    c.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
