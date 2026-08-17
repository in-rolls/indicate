# Changelog

## 0.9.0 — 2026-08-16

### Added

- Bengali-to-English is now a local lookup direction, including Eastern Nagari
  words that use Assamese `ৰ` and `ৱ`. Its compiled `lookup.tsv.gz` downloads
  from the pinned Hugging Face model-assets revision on first use. The source
  corpus stays with its collection and provenance pipeline; millions of CSV
  rows are not duplicated in this repository or the wheel.

### Fixed

- OpenAI Batch requests use `max_completion_tokens`. `max_tokens` is rejected
  by current models such as `gpt-5.4-mini`, which previously made every request
  in an otherwise valid batch fail before inference.
- Lookup-only pairs no longer pretend to have seq2seq weights. Status reporting,
  artifact checks, and `indicate info` now ask for model files only when the
  direction actually supports the model backend.
- Model assets are pinned to an immutable Hugging Face commit rather than a
  moving branch or tag.

## 0.8.0 — 2026-08-14

### Changed — breaking

- **One transliteration API.** `indicate.transliterate(text, source=..., target=...)`
  and `transliterate_batch` replace `indicate.hindi2english` / `punjabi2english`
  and the `HindiToEnglish` / `PunjabiToEnglish` classes, which are **removed**
  along with their modules. The language is an argument, and is auto-detected
  from the script when omitted — feeding Gurmukhi to `hindi2english` used to
  produce garbage silently.
- **Backends are an ordered chain, not a boolean.** `engine=["lookup", "model"]`
  (the default) replaces `lookup=True/False` and the `LOOKUP_ENABLED` class
  attribute. `["model"]`, `["lookup"]`, `["lookup", "llm"]` and
  `["lookup", "model", "llm"]` are now all expressible; the last two could not
  be said before. A word goes to the first backend that will answer it.
- **One CLI command.** `indicate transliterate` replaces `hindi2english`,
  `punjabi2english` and `llm`; `--from/--to/--engine` replace the command name
  and `--lookup/--no-lookup`. New: `indicate languages`. The file-safety options
  (`--backup`, `--atomic`, `--dry-run`, `--format json`) now apply to every
  backend rather than only the LLM path.
- `indicate.batch` takes `engine=` instead of `lookup=`, and runs the whole
  non-LLM prefix of the chain locally before submitting, so
  `("lookup", "model", "llm")` submits only what both decline.
- `--resume` and `--show-examples` are gone from the CLI. Durable resumption
  lives in `indicate.batch`, which checkpoints every resolved token; a second
  resume mechanism on the file path was a second source of truth.

### Added

- `indicate/languages.py` — one registry of languages, aliases, scripts and
  local model pairs, replacing four overlapping tables.
- `indicate/engine.py` — the `Backend` protocol and the `lookup` / `model` /
  `llm` backends. An empty candidate list now counts as *decline*, so a failed
  decode falls through to the next backend instead of emitting an empty string.
- `indicate.supported()` / `supports()` — what this install can do, per backend.
  An unsupported direction raises `UnsupportedPairError` naming what would work,
  rather than silently falling through to a backend that costs money.
- A failing LLM call declines rather than raising, so `("lookup", "llm", "model")`
  degrades to local decoding.
- `python -m indicate` works, not only the installed `indicate` console script.
- **`INDICATE_DATA_DIR`** — a directory searched before the packaged one, for
  both lookup tables and weights. Without it an installed package could only
  read from `site-packages/indicate/data/` while `training/build_lookup.py`
  wrote into a checkout, so the documented "build your own table" instruction
  ended at a file the package would never open. The builder now writes there
  too, making it a working two-step:

  ```bash
  export INDICATE_DATA_DIR=~/.local/share/indicate
  uv run --group train python training/build_lookup.py --lang punjabi
  ```

### Fixed

- **A fresh install used to fail silently.** With no weights, no lookup table and
  no network, every backend declined and the API returned `""` — a plausible
  transliteration of nothing — while the CLI printed a blank line and exited 0.
  Now `BackendsUnavailableError` is raised, naming each backend and what to do
  about it, and the CLI exits 1. Declining is still silent: `engine=["lookup"]`
  over an uncovered corpus legitimately answers nothing.
- **`engine=["lookup"]` returned `""` for every installed user.** The table was
  never uploaded to Hugging Face, so the fetch 404'd on any machine that had not
  built one. `lookup.DOWNLOADABLE` now states which tables are actually
  published (currently none — they derive from corpora that are not ours to
  redistribute), so the lookup backend is reported as `unavailable` instead of
  being advertised, and no install pays a guaranteed-404 round trip.
- `indicate languages` reports per-backend status (`ready` /
  `downloads on first use` / `unavailable`) rather than listing directions it
  cannot actually serve.
- Writing the output over the input file is refused again; the path guard was
  reachable only when an input path was resolved, which the new CLI did not do.
- The model backend declines instead of crashing when the weights cannot be
  obtained at all, not only when decoding fails.
- **`indicate.batch` aborted on a stock install.** With the default
  `("lookup", "llm")` chain and no lookup table — the state every installed user
  is in — the local prefix reported itself unavailable and
  `submit_transliteration_batches` raised instead of submitting to the provider.
  An unavailable local prefix now means "nothing resolved locally", which is
  what the `llm` suffix is for.
- **A typo in the engine chain billed you.** `("lookpu", "llm")` was swallowed
  as "no local backend here", after which every token was submitted to a paid
  provider. The whole chain is validated before any local resolution, so a
  misspelling costs an exception rather than money.
- `transliterate_batch` raised `UnsupportedPairError` on input it supports when
  the first 50 entries were blank: detection sampled the first 50 *positions*
  rather than the first 50 non-blank texts.
- `indicate languages` reported the model `ready` when only one of the two
  weight files was present, e.g. after an interrupted download.
- Three test-infrastructure false greens: a failed `uv build`, a failed wheel
  install, and a deleted Hugging Face revision each turned into a skip, so the
  jobs added to catch exactly those failures could pass while they happened.
- **A chain with no `llm` still submitted to a paid provider.**
  `engine=("lookup",)` — the documented "is my corpus already covered?" probe —
  answered the hits from the table and then sent every miss to the provider.
  It now resolves locally and submits nothing.
- A lookup table that is valid gzip but invalid UTF-8 raised out of
  `LookupBackend.resolve` instead of disabling itself, taking the model
  fallback down with it.
- `--dry-run --output missing/dir/out.txt` created `missing/dir/`. A command
  that promises to write nothing now writes nothing.
- The `live-hf` job was unreachable: `e2e.yml` had no `tags:` trigger, so the
  job guarded on `refs/tags/` had never run — including the Hugging Face
  contract check that exists to catch a missing release asset.
- `gazetteer/harvest_wikidata.py` paged SPARQL results with `LIMIT`/`OFFSET`
  and no `ORDER BY`, which can duplicate or skip rows between pages.
- Tests can no longer reach a real provider: `tests/conftest.py` replaces every
  provider credential with an obvious fake for the whole suite.
- **A local backend placed after `llm` never ran.** `engine=("lookup", "llm",
  "model")` resolves only the prefix before `llm` locally, so words the provider
  missed went straight back to the provider instead of to the free local
  decoder the caller had asked for. Backends after `llm` now run before any
  requeue is paid for.
- **Resuming a batch with new tokens bypassed the table.** With a state file
  present the driver skipped submission entirely, and with it the local prefix,
  so tokens added on the resume went straight to the paid requeue. They are now
  offered to the local backends first.

### Testing

- **The wheel is now tested, not the source tree.** `tests/e2e/` builds the
  wheel, installs it into a clean venv with only `click`, and asserts
  `indicate.__file__` resolves under site-packages — CI's old wheel job ran
  pytest from the repo root, where `import indicate` found `./indicate/`. It
  also pins what the wheel must *not* contain (weights, lookup tables, corpora)
  with a size ceiling, and exercises the console script as a real process.
- **A fresh clone is a supported configuration with its own assertions.**
  `tests/conftest.py` detects artifacts offline, marks tests `needs_lookup` /
  `needs_weights`, and prints what was missing with the command that builds it.
  `--require-artifacts` turns those skips into failures; CI runs one leg with it
  after building both tables from the committed corpora.
- Replaced the tests that could not fail: five invoked the deleted
  `hindi2english` command and passed vacuously, and a family of them wrapped the
  assertion in `try/except: assertIsInstance(e, Exception)`.
- New: `tests/test_engine_availability.py` (declined vs unavailable),
  `tests/test_resources.py` (the exact Hugging Face path join, and that a local
  file means no download), `tests/test_cli_plumbing.py` (`--format json`,
  `--backup`, `--dry-run`, `--no-atomic`, `--n`, blank-line alignment, with the
  backends stubbed so it needs no artifacts), `tests/e2e/test_hf_contract.py`
  (`live`-marked; asserts the model repo holds every file the loader requests).

### Verified

- The refactor is **byte-identical** to the previous implementation on 20,000
  outputs (5,000 roll strings and 2,500 Dakshina words per language, both arms).

## 0.7.0

### Added
- **Data-scaled v2 models** — both Hindi and Punjabi retrained on the public
  **Aksharantar** corpus (AI4Bharat) merged with our own data: Hindi ~1.5M pairs,
  Punjabi ~773k. A strict **leakage filter** drops every eval word from training.
- **Batched inference** — `transliterate_batch(list[str])` is the decode engine
  (batched greedy + beam); `transliterate(str)` is a thin wrapper. The CLI
  `--batch` mode and bulk eval use it (≫ faster than one-at-a-time).
- **Top-k / n-best** — `transliterate(text, n=k)` returns up to `k` ranked
  candidates (a `list[str]`); `n=1` (default) still returns a `str`.

### Changed
- **Model weights hosted on Hugging Face** (`soodoku/indicate`, tag `v0.7.0`) and
  lazy-downloaded (cached) on first use, so the wheel is ~50 KB instead of ~107 MB
  (PyPI's 100 MB limit). Tokenizer JSONs still ship in the package; a local copy of
  the weights, if present, is used in preference to the download. Weights are fp32
  (`safetensors`) — fp16 was tried and rejected (it perturbed ~0.05% of outputs for
  no size benefit worth keeping, since hosting solves the size problem).
- **Input-adaptive decode cap** (`min(max_length_output, 2·len(input)+8)`).
- Removed `func_timeout` (the decoder is hard-bounded, so the wall-clock guard
  was dead weight — and a perf/flakiness drag); dropped the dependency.
- Coverage reports to the terminal only (no `htmlcov`/`coverage.xml`).

### Accuracy (Dakshina test, exact-match; vs AI4Bharat IndicXlit, same direction/data)
| Model | Dakshina (gold) | Held-out-own names | 
|-------|-----------------|--------------------|
| Hindi v2 | **74.4%** (IndicXlit 73.2%) | **52.8%** (IndicXlit 49.7%) |
| Punjabi v2 | 71.9% (IndicXlit 73.2%) | **56.9%** (IndicXlit 53.5%) |

v2 matches/edges SOTA IndicXlit on the gold benchmark and **beats it on the
deployment domain** (names IndicXlit never trained on). Eval is leakage-filtered;
see `training/IMPROVEMENTS.md` and `training/compare.py`.

## 0.6.0

### Added
- **Punjabi (Gurmukhi → English) transliteration model** — new `PunjabiToEnglish`
  class, `indicate.punjabi2english(...)` API, and `indicate punjabi2english` CLI
  command. Distilled from GPT-4o-labelled Punjab data
  (`training/extract_punjabi.py` → `data/punjabi.csv.gz`).
- **Beam-search decoding** (length-normalized, width 5) is now the default for
  both models — +1.5 pts exact-match on the Hindi Dakshina test set vs greedy.
- Evaluation tooling: `training/eval.py` (gold Dakshina), `training/oos_eval.py`
  (held-out), and `training/metrics.py` reporting exact-match accuracy, CER, and
  Acc@≤1 with edit-distance histograms.

### Changed
- **Migrated the modeling stack from TensorFlow to PyTorch** and the package to
  **Python 3.13-only**. Weights now ship as `safetensors`.
- Refactored the per-language logic into a shared `Seq2SeqTransliterator` base.

### Accuracy (Google Dakshina test, exact-match / Acc@≤1 / CER)
- Hindi → English: **77.32% / 91.16% / 5.65%**
- Punjabi → English: **71.24% / 91.56% / 6.42%**

### Notes
- Char-LM beam re-ranking and attention padding masks are implemented but gated
  off (no measured win for the name-heavy, native→Latin case); see
  `training/IMPROVEMENTS.md`. The next milestone is a data-scale retrain on the
  public Aksharantar corpus.
