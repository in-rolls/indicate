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

## J&K Hindi review handoff

The September 12, 2026 instate review compares 5,857 selected native tokens with
this Hindi corpus and `../eroll_transliteration/data/hindi.csv.gz`. A shared Latin
candidate is evidence of overlap, not necessarily a unique answer. Of 2,838
matching candidate sets, 1,321 have a single matching candidate and 1,517 contain
alternatives. Neither figure measures transliteration accuracy.

The completed Muse Spark 1.3 Contributor diagnostic covers all 5,857 tokens:
3,031 primary answers match the historical eroll candidate, 605 list it as an
alternative, 542 differ and 1,679 have no eroll candidate. Responses flag 503
unusual source spellings and 33 uncertain readings. These comparisons are not
verified transliteration accuracy and must not become training labels or test
references without independent validation.

Instate's candidate map retains 3,503 native forms, covering 913,449 selected
occurrences in the corrected handoff. It withholds source warnings, insufficient
corroboration and rows that fail its local rule requiring at least two ASCII
letters in every returned Latin form. The high-frequency `सिहं` remains withheld:
the response gave literal `Sihan` and alternative `Singh`, while font checks
confirm the inspected source sequence as written. The intended conventional name
is unresolved. A historical translation such as `बेसिन` → `washbasin` also shows
why a corpus candidate is insufficient as a name reference.

The corpus and model artifacts are unchanged. The prompt, responses, row
comparison, candidate map and cost record are retained in the sibling instate
workspace under `data/jk_recovery/muse_review/`; `hindi_full/summary.json` records
the complete diagnostic. The full candidate handoff preserves native selections
and evidence exactly; its linguistic accuracy remains unmeasured.

A separate check uses the original, human-annotated Google Dakshina Hindi
lexicons. All three splits were verified against the publisher's archive; the
existing local test file matches exactly. Of the retained candidate's 3,503 native
forms, 1,103 have an exact attested Latin alternative, covering 856,370 occurrences.
Fifty-nine forms covering 1,096 occurrences have other attested alternatives; 2,341
forms covering 55,983 occurrences have no reference entry. Missing exact matches
are review cases, not automatic errors. The lexicons are not exhaustive, their
Wikipedia overlap is not representative of J&K names, and model training exposure
is unknown. This is lexical support, not untouched test accuracy. See instate's
`data/jk_recovery/muse_review/hindi_full/dakshina_support_summary.json` and the
[dataset documentation](https://github.com/google-research-datasets/dakshina).
The source corpus, candidate map and model artifacts remain unchanged.

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

A convention review of the original 60 differing reference cases passed all 12
synthetic controls. It classified 59 as compatible conventions and one as
uncertain. The reviewed instate candidate changes only `बख्शी` from `bakshi` to
`bakhshi`, supported by both a human-attested alternative and the earlier blind
model answer. Exactly 25 Latin-name occurrences change; native selections,
abstentions and household/relative evidence remain unchanged. This is a reviewed
candidate map change, not a source-corpus or model update. The current handoff is
`data/jk_recovery/hindi_reviewed_candidate/` in instate.

## J&K Urdu handoff

Instate's final calibrated 2018 Urdu recovery preserves 4,608,102 active assembly
records. Upnaam selects 970,947 corroborated native surname occurrences from
4,399 distinct selected tokens and abstains on 3,637,155 records. The source
recovery matched 1,259 of 1,322 hidden word controls exactly (95.23%); this
measures transcription, not surname or pronunciation accuracy.

The shared lookup corpus remains in instate at
`data/jk_recovery/muse_review/urdu_all_tokens/`:

- `urdu.csv.gz` contains eligible native/Latin pairs;
- `transliterations.parquet` records completed decisions, alternatives, evidence
  and exclusions;
- `provenance.json` records review batches and the selection policy; and
- `validation.json` records corpus hashes and selected-handoff coverage.

The corpus contains 45,427 reviewed token types and 27,221 eligible pairs. The
final 164-type selected-token review adds 1,450 mappings to the prior 25,771-pair
map. All 4,399 selected Urdu types are mapped, covering all 970,947 selected
occurrences. Across all accepted decoded own and relative names, eligible pairs
cover 5,581,174 of 6,168,112 token occurrences (90.48%). Coverage does not measure
transcription or pronunciation accuracy.

Build the installed table with:

```sh
python training/build_lookup.py --lang urdu \
  --corpus ../instate/data/jk_recovery/muse_review/urdu_all_tokens/urdu.csv.gz
```

The resulting `lookup.tsv.gz` has 27,221 keys, no contested keys and no ties. Its
SHA-256 is
`570206fa75d4c12169bd7d09aa081ce0b268f8208d50fa7c5aaa622c5f862950`.
The corpus SHA-256 is
`35be72079f8a226db1e9794ac91fa3c9f921fc673287c93450f23514502cd373`.
No corpus copy is stored in indicate, and no API call occurs during lookup use.
Model-only mappings remain silver annotations; quarantined, warned and unsupported
decisions stay outside the table.
