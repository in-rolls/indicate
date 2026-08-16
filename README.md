# Indicate: Transliterate Indic Languages with PyTorch and LLMs

[![PyPI Version](https://img.shields.io/pypi/v/indicate.svg)](https://pypi.python.org/pypi/indicate)
[![Downloads](https://static.pepy.tech/badge/indicate)](https://pepy.tech/project/indicate)
[![Tests](https://github.com/in-rolls/indicate/workflows/CI/badge.svg)](https://github.com/in-rolls/indicate/actions?query=workflow%3ACI)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://in-rolls.github.io/indicate/)

**Indicate** provides high-quality transliteration between Indic languages and English using both a traditional PyTorch model and state-of-the-art LLMs (Large Language Models).

## 🚀 Features

- **🔀 Composable Backends**: Chain a word table, a local model and an LLM in any order
- **🌍 Multi-Language**: 12+ Indic languages, with the source script auto-detected
- **🔄 Bidirectional**: Supports both Indic→English and English→Indic transliteration
- **🛡️ Production Ready**: Safe file handling, atomic writes, backup support
- **📊 Structured Output**: Rich JSON format with metadata and error handling
- **⚡ Batch Processing**: Efficient processing of large files with progress tracking

## 🎯 Supported Languages

Hindi • Tamil • Telugu • Bengali • Gujarati • Kannada • Malayalam • Punjabi • Marathi • Odia • Urdu • Sanskrit ↔ English

## Install

We strongly recommend installing `indicate` inside a Python virtual environment (see [venv documentation](https://docs.python.org/3/library/venv.html#creating-virtual-environments))

**Requirements:** Python 3.13+

```bash
pip install indicate
```

## 🔧 Quick Setup

### For LLM-based transliteration (recommended):
```bash
pip install indicate

# Set your API key (choose one):
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key  
export GOOGLE_API_KEY=your-key
```

### For the local model (no API key):
```bash
pip install indicate
# No API key needed. The PyTorch weights are downloaded once from Hugging Face
# (gojiberries/indicate) on first transliterate and cached locally; tokenizers ship
# in the wheel. After the first run it works fully offline.
```

### For the lookup backend (optional, and worth it):

The word table **does not ship in the wheel and is not downloadable**. It is
derived from `data/hindi.csv.gz` (which blends CC-BY-NC IIT Bombay pairs) and
`data/punjabi.csv.gz` (from a restricted electoral-roll deposit), neither of
which is ours to redistribute under MIT. Build it from a checkout in a few
seconds:

```bash
export INDICATE_DATA_DIR=~/.local/share/indicate     # where your tables live
uv run --group train python training/build_lookup.py --lang hindi
uv run --group train python training/build_lookup.py --lang punjabi
```

`INDICATE_DATA_DIR` is where the builder writes and where an installed package
looks first. Without it the table lands inside the checkout, which a
`pip install`ed copy in `site-packages` will never read. Keep it exported and
`indicate languages` flips that row from `unavailable` to `ready`:

```
Direction                 Backend   Status
punjabi -> english        lookup    ready
                          model     ready
```

Without a table nothing breaks — `lookup` declines every word and `model`
answers them.

## 🎯 Usage

One command, one function. The language and the backend are arguments, not
separate entry points.

```bash
# Source language auto-detected from the script
indicate transliterate "राजशेखर चिंतालपति"
# rajshekhar chintalpati

indicate transliterate "ਰਵਿ ਸ਼ਰਮਾ"
# ravi sharma

# Devanagari carries several languages and detection picks Hindi, so say it
# explicitly when it is not. Marathi has no local model — hence --engine llm
indicate transliterate "नमस्ते" --from marathi --engine llm

# Files, with the usual safety options
indicate transliterate --input names.txt --output roman.txt --format json --backup
indicate transliterate --input names.txt --output roman.txt --dry-run

# What can this install actually do?
indicate languages

# Model architecture, training sources, where the weights come from
indicate info
```

`python -m indicate` does the same as the `indicate` script, for when the
console script is not on `PATH`.

```python
import indicate

indicate.transliterate("राजशेखर चिंतालपति")  # "rajshekhar chintalpati"
indicate.transliterate("ਰਵਿ", source="punjabi")  # "ravi"
indicate.transliterate("नमस्ते", n=3)  # 3 ranked candidates
indicate.transliterate_batch(["हिंदी", "मुंबई"])  # ["hindi", "mumbai"]

indicate.supported()  # {(source, target): (backends...)}
```

### Choosing the engine

A word is answered by the first backend that will answer it. The chain is an
argument, so you decide how much machinery each word is worth:

| chain | what it does |
|---|---|
| `lookup, model` | **default** — read the table, decode the rest locally |
| `model` | decode everything; what a benchmark must use |
| `lookup` | table only, `""` on a miss — "is my corpus already covered?" |
| `lookup, llm` | the table intercepts the paid path |
| `lookup, model, llm` | escalate to a provider only what both decline |
| `llm` | ask a provider for everything |

```bash
indicate transliterate "मुंबई" --engine model
indicate transliterate "मुंबई" --engine lookup,llm --provider openai
```

```python
indicate.transliterate("मुंबई", engine=["lookup", "llm"])
indicate.transliterate("मुंबई", engine="model")
```

A backend that cannot serve a direction is skipped; if none remain you get an
error naming what would work, rather than a silent fallback onto something that
costs money:

```
$ indicate transliterate "வணக்கம்"
Error: no backend in ['lookup', 'model'] supports tamil->english;
try engine=['llm'] or see indicate.supported()
```

That is `UnsupportedPairError`. A different failure gets its own type, because
the two mean opposite things:

- a backend that **declined** — it loaded its table and had no entry for that
  word — is ordinary and silent. `engine=["lookup"]` over an uncovered corpus
  declines everything and returns `""`, which is the whole point of asking.
- a backend that was **unavailable** — no table built, no weights, no network —
  answers nothing because it could not run. When *every* backend in the chain is
  in that state you get `BackendsUnavailableError` naming each one and what to
  do about it, rather than an empty string that looks like an answer.

```python
try:
    indicate.transliterate("राजशेखर")
except indicate.BackendsUnavailableError as exc:
    print(exc)  # nothing could answer 1 word(s): lookup has no table (build ...
```

### Why the lookup backend is first by default

Known words are answered from the word table and never reach the decoder. On
Punjab electoral-roll text that covers 99.1% of tokens, so the model handles the
tail: **42x** the end-to-end throughput (10,937 tok/s against 258), and an input
that hits entirely never even imports torch, which is worth **4.4x** on cold
start (0.10s to first answer against 0.44s). `training/bench_lookup.py`
reproduces both.

It is also more accurate than either component alone, because the builder
declines to answer where the training corpus has no majority and lets those
words fall through: on the Dakshina test set, 78.8% exact against the model's
76.2% for Hindi, 77.6% against 77.0% for Punjabi.

Two caveats worth knowing before you rely on those numbers. They are measured on
**electoral-roll names**; on general Wikipedia prose the same table covers 56.9%
of tokens, not 99.1%, and the cold-start win largely disappears because a
sentence almost always contains a miss. And the shipped table contains 908 of
the 2,500 Dakshina Hindi test words, so the Hindi accuracy figure is optimistic
by an unknown amount. `training/build_lookup.py --eval-clean` builds a table
with every eval word withheld.

Use `--engine model` (or `engine=["model"]`) to measure the model by itself —
benchmarks must, or they score memorization. `training/seam_check.py` checks
that mixing table and model output in one string stays stylistically consistent.

### The LLM backend directly

For whole-sentence transliteration with context, use the client rather than the
engine chain — the chain resolves word by word:

```python
from indicate import IndicLLMTransliterator

transliterator = IndicLLMTransliterator("hindi", "english")
transliterator.transliterate("राजशेखर चिंतालपति")
transliterator.transliterate_batch(["राजेश", "गौरव", "प्रिया"])
```

For millions of tokens, `indicate.batch` submits to a provider's async Batch API
with checkpointing, and answers what it can locally first:

```python
from indicate.batch import transliterate_tokens_batched

pairs = transliterate_tokens_batched(
    tokens,
    "punjabi",
    "english",
    checkpoint_path="run.jsonl",
    engine=("lookup", "llm"),  # default; ("lookup","model","llm") goes further
)
```

## 📊 JSON Output Format

`--format json` works with every backend, not just the LLM. One line of input in,
one entry out, with the chain that answered it recorded per row:

```json
{
  "metadata": {
    "source_language": "hindi",
    "target_language": "english",
    "timestamp": "2026-08-14T07:40:08.697757+00:00",
    "total_lines": 1,
    "successful_lines": 1,
    "failed_lines": 0,
    "format_version": "1.0",
    "encoding": "utf-8",
    "description": "Indic language transliteration results from indicate package"
  },
  "results": [
    {
      "line_number": 1,
      "input_text": "राजेश कुमार",
      "output_text": "rajesh kumar",
      "source_lang": "hindi",
      "target_lang": "english",
      "confidence": "lookup,model",
      "error": null,
      "processing_time": 0.07029390335083008,
      "timestamp": "2026-08-14T07:40:08.697423+00:00"
    }
  ]
}
```

`confidence` holds the engine chain, not a probability — the local model's beam
scores are not calibrated, so publishing one would invite a comparison it cannot
support.

## 🛡️ Safety Features

- **🔒 Input/Output Validation**: Prevents accidental file overwrites
- **⚛️ Atomic Writing**: Safe file operations using temporary files
- **💾 Automatic Backups**: Optional timestamped backups of existing files
- **👁️ Dry Run Mode**: Preview operations before execution

Resumable runs live in `indicate.batch`, which checkpoints every resolved token
to disk and picks up where it left off.

## 🎛️ Advanced Usage

```bash
# Pick an LLM provider and model
indicate transliterate "text" --engine llm --provider anthropic --model claude-3-opus

# Read JSON produced by an earlier run
indicate transliterate --input results.json --from english --to hindi --engine llm

# Table only: how much of this file does the table already cover?
indicate transliterate --input names.txt --engine lookup
```

## 🔄 Backend Comparison

| | `lookup` | `model` | `llm` |
|---|---|---|---|
| **Directions** | Hindi, Punjabi → English | Hindi, Punjabi → English | 12+ languages, any Indic pair |
| **Setup** | build a table (one command) | none | API key |
| **Speed** | 10,937 tok/s end to end | 258 tok/s | network-bound |
| **Cost** | free | free | per API call |
| **Offline** | ✅ | ✅ | ❌ |
| **Coverage** | only what is in the table | every word | every word |
| **Answers with** | the corpus label | a decode | the provider |

Both speeds are end-to-end on roll names, measured back to back on one machine,
so the ratio is the meaningful part. The table itself serves 16.9M reads/s once
loaded; that number describes the dictionary, not the pipeline, and quoting it
as throughput would overstate the win by three orders of magnitude.

`indicate languages` prints which of these are available for a direction on your
machine.

## 🧪 Testing Locally

1. **Clone and install**:
   ```bash
   git clone https://github.com/in-rolls/indicate.git
   cd indicate
   uv sync  # or pip install -e .
   ```

2. **Run tests**:
   ```bash
   uv run pytest                       # everything
   uv run pytest tests/test_engine.py  # one file
   ```

   Model weights and lookup tables are gitignored, so a fresh clone skips the
   tests that need them and prints what is missing with the command that builds
   it. To make those skips into failures instead — which is what CI does, after
   building the tables from the committed corpora:

   ```bash
   uv run pytest --require-artifacts
   ```

3. **Test the backends**:
   ```bash
   # Local, no API key
   indicate transliterate "हिंदी" --engine lookup,model

   # LLM (set an API key first)
   export OPENAI_API_KEY=your-key
   indicate transliterate "हिंदी" --engine llm
   ```

## Data

The datasets used to train the model:

- [Indian Election affidavits](https://affidavit.eci.gov.in/CandidateCustomFilter)
- [Google Dakshina dataset](https://github.com/google-research-datasets/dakshina)
- [ESPN Cric Info](https://www.espncricinfo.com/hindi/series/pakistan-tour-of-england-2021-1239529/england-vs-pakistan-1st-odi-1239537/full-scorecard) for hindi version of the [english scorecard](https://www.espncricinfo.com/series/pakistan-tour-of-england-2021-1239529/england-vs-pakistan-1st-odi-1239537/full-scorecard)
- [IIT Bombay English-Hindi Corpus](https://www.cfilt.iitb.ac.in/iitb_parallel/)

## Evaluation

The v2 models (trained on our data + the public [Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar)
corpus) are benchmarked against **AI4Bharat IndicXlit** — the same direction
(native→Latin), the same test sets, the same metric (Top-1 exact-match,
match-any-reference). Training is leakage-filtered so no eval word appears in it.

| Model | Dakshina (gold) | Held-out-own names¹ |
|-------|-----------------|---------------------|
| Hindi → English | **74.4%** (IndicXlit 73.2%) | **52.8%** (IndicXlit 49.7%) |
| Punjabi → English | 71.9% (IndicXlit 73.2%) | **56.9%** (IndicXlit 53.5%) |

¹ Held-out slice of our own electoral/affidavit names — the cleanest comparison,
since IndicXlit never trained on it. **v2 matches or edges IndicXlit on the gold
benchmark and beats it on the deployment domain.** Primary metric is Top-1
exact-match; CER (character error rate) is the soft companion. Reproduce with
`training/eval.py` and `training/compare.py`.

Below is the edit-distance distribution on the test set (0 = exact match):

![Edit distance metrics of model on Google Dakshina test dataset](https://github.com/in-rolls/indicate/raw/master/images/h2e_ed.png)

## Authors

Rajashekar Chintalapati and Gaurav Sood

## Contributor Code of Conduct

The project welcomes contributions from everyone! In fact, it depends on it. To maintain this welcoming atmosphere, and to collaborate in a fun and productive way, we expect contributors to the project to abide by the [Contributor Code of Conduct](http://contributor-covenant.org/version/1/0/0/).

## License

The package is released under the [MIT License](https://opensource.org/licenses/MIT).
