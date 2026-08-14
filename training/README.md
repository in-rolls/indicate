# Training the transliteration models

This directory holds the PyTorch training, extraction, and evaluation scripts for
the encoder-decoder (LSTM + Luong attention) transliteration models shipped in
`indicate/` — currently **Hindi** (Devanagari) and **Punjabi** (Gurmukhi), both
→ English.

The same `train.py` trains either model; `--model-dir` selects where the
tokenizers and `saved_weights/` live.

## Hindi

Training uses the committed parallel corpus `data/hindi.csv.gz` (371k Hindi/English
character pairs from Indian election affidavits, ESPN Cricinfo, the Google
Dakshina dataset, and the IIT Bombay corpus). **No download is required** — the
merged corpus is already in the repo. By default the existing tokenizer JSONs are
reused (so vocabulary indices stay stable); pass `--rebuild-vocab` to refit.

```bash
# Full run (auto-selects cuda > mps > cpu)
python training/train.py --epochs 25 --batch-size 64

# Quick smoke test on a subset
python training/train.py --limit 5000 --epochs 1
```

## Punjabi

The Punjabi pairs are distilled from GPT-4o. The raw source
`data/punjab_transliteration_subset.parquet` (Punjab electoral rolls, ~19M rows,
**not committed**) has each Gurmukhi run transliterated in place, so aligned word
pairs can be recovered:

```bash
# 1. Extract unique (gurmukhi -> english) word pairs -> data/punjabi.csv.gz (committed)
python training/extract_punjabi.py

# 2. Train (own Gurmukhi + English vocabularies)
python training/train.py \
    --data data/punjabi.csv.gz \
    --model-dir indicate/data/punjabi_to_english \
    --input-vocab-name punjabi_tokens.json --target-vocab-name english_tokens.json \
    --max-input 32 --max-output 32 --rebuild-vocab \
    --epochs 25
```

Best weights (lowest validation loss) are written to
`<model-dir>/saved_weights/{encoder,decoder}.safetensors`.

## Evaluate

The canonical benchmark uses the **Google Dakshina** romanization lexicons. The
2 GB dataset is not committed; download it once
(https://github.com/google-research-datasets/dakshina, release tarball
`dakshina_dataset_v1.0.tar`) and place the per-language test splits under
`data/dakshina/`:

- Hindi:   `dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv`
- Punjabi: `dakshina_dataset_v1.0/pa/lexicons/pa.translit.sampled.test.tsv`

```bash
python training/eval.py --model hindi      # data/dakshina/hi.translit.sampled.test.tsv
python training/eval.py --model punjabi    # data/dakshina/pa.translit.sampled.test.tsv
```

A native word counts as correct if the prediction matches any reference
romanization (the test set lists multiple valid spellings per word).

**v2 (Aksharantar-scaled) vs AI4Bharat IndicXlit** — same direction, test sets,
and metric (Top-1 exact-match). Training is leakage-filtered (`build_v2.py`).

| Model | Dakshina (gold) | Held-out-own names |
|-------|-----------------|--------------------|
| Hindi → English | **74.4%** (IndicXlit 73.2%) | **52.8%** (IndicXlit 49.7%) |
| Punjabi → English | 71.9% (IndicXlit 73.2%) | **56.9%** (IndicXlit 53.5%) |

Held-out-own is the cleanest comparison (IndicXlit never trained on our
electoral/affidavit names). Reproduce the baseline with `baseline_indicxlit.py`
(isolated env) + `score_preds.py`, or the full breakdown with `compare.py`.
`oos_eval.py` reports held-out accuracy on the training corpus itself.

Every eval script passes `engine=("model",)`, so these numbers are the model alone.
With the lookup backend in the chain they would partly measure the table's memory of the
corpus, which is not what a model benchmark should report.

## Lookup table

```bash
python training/build_lookup.py --lang punjabi        # -> indicate/data/<lang>_to_english/lookup.tsv.gz
python training/build_lookup.py --lang hindi --eval-clean   # excludes eval-set source words
python training/bench_lookup.py --lang punjabi        # latency, coverage, cold start
python training/seam_check.py  --lang punjabi         # does splicing table + model output show?
```

The table answers known words before the decoder sees them. It is rebuilt from
the committed corpora and is byte-identical on a rebuild, so a change in the
output means a change in the corpus or in the build rule.

| | keys | roll token mass | ties left to the model |
|---|---|---|---|
| Punjabi (`data/punjabi.csv.gz`) | 286,546 | 99.12% | 109 |
| Hindi (`data/hindi.csv.gz`) | 209,925 | — | 25,199 |

Hindi drops 10.7% of its keys because its corpus is phrase-level: positional
alignment gives one attestation per pair, so `मुंबई` is attested once each as
`mumbai`, `mumabi`, `mumba` and `mumbaikar`. Guessing among those cost more than
it gained, so a tie is not answered at all. Use `--eval-clean` to build a table
with every eval-set source word withheld; `tests/test_build_lookup.py` asserts
no Dakshina test word survives in it.
