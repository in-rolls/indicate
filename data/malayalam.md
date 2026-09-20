# Malayalam word-pair corpus

`malayalam.csv.gz` holds 400,506 unique native keys with English spellings. It is a byte-identical copy of `eroll_transliteration/data/malayalam.csv.gz` after validation on 2026-09-07. SHA-256: `8601f940f21d97f887de9b9b11701dd4b6da3cf5667aad6a17e640a2b579f7f5`.

The corpus combines Kerala parallel electoral-roll spellings, the Lakshadweep SIR 2002 spelling harvest, and 26,606 validated tokens from the 2026 Malayalam OCR residue. These are spelling pairs, not person-level electoral records. Some native tokens are OCR fragments; their presence does not establish that they are real names or correctly transcribed text.

The 2026 harvest used gpt-5.4-mini-2026-03-17 in 1,065 batch requests. Provider-reported usage was 485,726 input and 185,703 output tokens, with no reasoning tokens or truncated requests, priced at $0.599979 at OpenAI batch rates. Five malformed outputs were corrected locally. The punctuation-only token `്്` was excluded. `data/malayalam.provenance.json` retains the usage audit, correction ledger and hashes of the raw batch artifacts. The raw provider journal stays local and is excluded from the repository.

Build the local table with `uv run --group train python training/build_lookup.py --lang malayalam`. The build yields 399,512 keys after normalization and alignment filtering; 12 tied keys are withheld. The table records its corpus hash. Atomic chillu letters and their consonant/virama/ZWJ encodings share lookup keys. There is no Malayalam sequence model or downloadable lookup in the pinned model repository.

To refresh after another validated harvest, copy the corpus from eroll_transliteration, record its new hash and provenance here, rebuild the lookup, and run the build/lookup tests. Existing source terms still apply; the corpus and table are not included in the package wheel.
