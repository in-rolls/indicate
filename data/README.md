# Data

This directory holds **training & source data only** — it is excluded from the
installable package (`tool.uv.build-backend.source-exclude` drops `/data`), so
nothing here ships in the PyPI wheel. The runtime tokenizers (`*_tokens.json`)
ship in the wheel under `indicate/data/`; the **safetensors weights are hosted on
Hugging Face** (`soodoku/indicate`, tag matching the package version) and
lazy-downloaded/cached on first use — they are kept out of both the wheel
(`wheel-exclude`) and git tracking (`.gitignore` ignores
`indicate/data/*/saved_weights/`). A local copy under `indicate/data/.../saved_weights/`,
if present, is used in preference to the download (handy for training/dev).

## What lives where

**In the repo** (small, derived, reproducible):
- `hindi.csv.gz` — Hindi→English training corpus, 371k char-level pairs (gzipped)
- `punjabi.csv.gz` — Punjabi→English training corpus, 287k pairs (gzipped)
- All collection/processing code: `get_affidavits/`, `get_espncricinfo/`,
  `railway_stations/`, `wikipedia_interwiki/`, and the notebooks
  (`notebooks/Data_Preparation.ipynb`, `examples/punjab_transliteration.ipynb`)

**On Dataverse** (source data, for reproducibility):
- **Hindi source** (public) — `affidavits.csv`, `players_with_hindi_names.json`,
  `en-hi.mined-pairs` (IIT Bombay, CC-BY-NC). New deposit DOI: `TODO`.
- **Punjab electoral rolls** (restricted: form + IRB) — *super-upstream* raw rolls.
  *Parsed Indian Electoral Rolls*, Sood et al., https://doi.org/10.7910/DVN/MUEGDT.
  File `punjab_all_clean+t13n.csv.gz` is the raw input to the LLM annotation.
- **Punjab LLM transliterations** (restricted) — `punjab_transliteration_subset.parquet`,
  the GPT-4o-annotated output (the direct source for `punjabi.csv.gz`). New
  restricted deposit DOI: `TODO`. Gitignored locally (201 MB, PII).

**Download from origin** (third-party, not redistributed):
- Google Dakshina benchmark → `data/dakshina/` (see `training/README.md`).
- **Aksharantar** (AI4Bharat, CC-BY/CC0) → fetched by `training/fetch_aksharantar.py`
  into `data/aksharantar/` (gitignored). The v2 data-scale source (Hindi 1.3M,
  Punjabi 515k pairs). https://huggingface.co/datasets/ai4bharat/Aksharantar

**Built locally, gitignored** (v2 — see `training/build_v2.py`):
- `data/<lang>_train_v2.csv.gz` — merged training corpus (ours + Aksharantar
  train/val), deduped, **leakage-filtered** so no training source appears in the eval.
- `data/eval/<lang>_blended.tsv` — blended multi-reference eval: Dakshina test +
  Aksharantar test + a held-out slice of our own corpus (union by source word).
  *Leakage filter is mandatory because Aksharantar contains Dakshina-sourced rows.*

## Pipelines (source → repo → model)

**Hindi**
```
[Dataverse: affidavits.csv, players_with_hindi_names.json, en-hi.mined-pairs]
  + Google Dakshina hi lexicon (download)
    → notebooks/Data_Preparation.ipynb        (merge, dedupe, clean)
    → data/hindi.csv.gz                        (committed)
    → training/train.py                        → indicate/data/hindi_to_english/
```

**Punjabi**
```
Parsed Indian Electoral Rolls (10.7910/DVN/MUEGDT, restricted)   [super-upstream raw]
  → examples/punjab_transliteration.ipynb     (extract Gurmukhi words, GPT-4o annotate)
  → punjab_transliteration_subset.parquet      [deposited: new restricted Dataverse]
  → training/extract_punjabi.py                (align word pairs, modal-dedupe)
  → data/punjabi.csv.gz                        (committed)
  → training/train.py                          → indicate/data/punjabi_to_english/
```

**v2 (Aksharantar-scaled)** — both languages
```
data/<lang>.csv.gz (ours) + Aksharantar train/val (download)
  → training/build_v2.py   (merge, dedupe, leakage-filter; build blended eval)
  → data/<lang>_train_v2.csv.gz + data/eval/<lang>_blended.tsv
  → training/train.py --rebuild-vocab → indicate/data/<lang>_to_english/
```

**Bengali lookup**
```
shared LLM-labeled Eastern Nagari elector-name corpus
  → deterministic, deduplicated lookup.tsv.gz
  → pinned Hugging Face model-assets revision
```

Only the compiled lookup is published. The multi-million-row source CSV stays
in `eroll_transliteration` with its collection and provenance pipeline; it is
not copied into this repository or package. The compiled table header records
the source corpus's SHA-256 for an exact, independently checkable build link.

Rebuild the published artifact without copying the corpus:

```bash
uv run --group train python training/build_lookup.py --lang bengali \
  --corpus ../eroll_transliteration/data/bengali.csv.gz
```

## Reproduce / download

```bash
# v1: train from the committed corpora (no download needed)
python training/train.py                                   # Hindi
python training/train.py --data data/punjabi.csv.gz \
    --model-dir indicate/data/punjabi_to_english --rebuild-vocab \
    --input-vocab-name punjabi_tokens.json --max-input 32 --max-output 32

# v2: scale with Aksharantar, then retrain + eval on the blended set
python training/fetch_aksharantar.py
python training/build_v2.py
python training/train.py --data data/hi_train_v2.csv.gz \
    --model-dir indicate/data/hindi_to_english --rebuild-vocab
python training/eval.py --model hindi --test-file data/eval/hi_blended.tsv
```
