# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Indicate is a Python package for transliterating Indic text to English using PyTorch-based encoder-decoder models with attention. It ships local models for **Hindi** (Devanagari) and **Punjabi** (Gurmukhi), plus an LLM backend for other languages. The local models are custom-trained neural networks with pre-trained weights.

## Development Commands

### Testing
```bash
uv run pytest                          # the whole suite
uv run pytest tests/test_engine.py     # one file
uv run pytest --require-artifacts      # fail, don't skip, on a missing artifact
uv run pytest -m "e2e and not live"    # build the wheel and exercise it
```

Weights and lookup tables are gitignored, so a fresh clone skips what needs
them and prints the build command in the terminal summary. `--require-artifacts`
turns those skips into failures; CI runs one leg that way. Do not use
`python -m unittest` — it collects roughly half the suite and errors.

### Build and Installation
```bash
uv sync           # Install dependencies with uv (recommended)
uv build          # Build package
pip install -e .  # Install in development mode (alternative)
```

### CLI Usage
```bash
# Modern Click-based CLI
indicate transliterate "राजशेखर चिंतालपति"
indicate transliterate --input file.txt --output result.txt --engine lookup,model
indicate languages
indicate info
```

### Documentation (Sphinx)
```bash
cd docs/ && uv run --group docs sphinx-build -W -b html . _build/html
```

## Architecture

### Core Components

1. **Seq2SeqModel** (`indicate/transliterator.py`) - one instance per language pair, built by `model_for(pair)` and cached in `_MODELS`; holds the vocab/weights paths and max lengths as instance state. torch is imported inside `load()`, not at module scope, so a run that never decodes never pays for it. `clear_models()` drops the cache.
2. **Pair registry** (`indicate/languages.py`) - languages, aliases, scripts, and the `(source, target)` pairs a local model exists for. Replaces the old class-per-language modules.
3. **Backends** (`indicate/engine.py`) - `lookup` / `model` / `llm` behind a `Backend` protocol, folded in order by `resolve_words`; each returns a candidate list per word or `None` to decline.
4. **Encoder** (`indicate/encoder.py`) - `nn.Module` LSTM encoder
5. **Decoder** (`indicate/decoder.py`) - `nn.Module` LSTM decoder with Luong (dot-product) attention
6. **Utils** (`indicate/utils.py`) - Tokenizer loading (`load_tokenizer`) and greedy decoding (`translate`)

### Model Architecture
- Encoder-decoder with Luong attention mechanism
- Embedding dimension: 256, LSTM units: 1024
- Per-language safetensors weights + tokenizer JSONs under
  `indicate/data/{hindi,punjabi}_to_english/`

### Training
- PyTorch training/extraction/eval scripts live in `training/` (see `training/README.md`)
- Hindi corpus `data/hindi.csv.gz` and Punjabi corpus `data/punjabi.csv.gz` are committed
  (Punjabi is extracted from the Dataverse-hosted parquet via `training/extract_punjabi.py`)
- Raw/large source data and the Dakshina benchmark live on Dataverse (see `data/README.md`)

### Data Pipeline
- Training data from ESPN Cricinfo, election affidavits, Google Dakshina dataset, and IIT Bombay corpus
- Character-level tokenization with special start (^) and end ($) tokens
- Decoding is hard-bounded by an input-adaptive step cap (no wall-clock timeout)

### Key Entry Points
- CLI: `indicate transliterate` (plus `languages`, `info`); `--from`/`--to`/`--engine`
- API: `indicate.transliterate(text, source=..., engine=[...])` and `transliterate_batch`
- Backends: `indicate/engine.py` (`lookup`, `model`, `llm`), chained in order; first to answer wins

## Dependencies
- Python 3.13+ (modern Python with enhanced type hints)
- Click 8.0+ (modern CLI framework)
- PyTorch 2.6+ (core ML framework)
- safetensors 0.4+ (model weight serialization)
- huggingface-hub 0.23+ (lazy weight download at first use)
- litellm 1.0+ (the `llm` backend; imported lazily, it costs ~1.3s)
- tqdm 4.60.0+ (progress bars)