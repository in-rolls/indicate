# Open Indic Romanization Gazetteer — corpus builder

Builds a ranked, provenance-tagged table of Indic-script tokens and their
attested romanizations. This directory is the **corpus builder**; it is not part
of the installed `indicate` package and ships in neither the wheel nor the sdist.

## Why this exists

There is no downloadable authoritative Indian romanization registry. Survey of
India holds "over 1.4 million ground-verified geographical names" whose field
workflow converted each one "from vernacular to Roman and Devanagari scripts"
([India's speaking notes to UNGEGN, 4th session,
2025](https://unstats.un.org/unsd/ungegn/sessions/4th_session_2025/documents/ND_3.4_Speaking_Notes_India.pdf)).
Those names are published — but *as labels on topographic maps*, not as a
joinable dataset. The pairs exist; the table does not. That format gap is what
this corpus fills.

So the corpus does not claim to be an official register. It claims to be
**what independent, openly-licensed sources attest, ranked, with provenance and
a confidence tier.**

## Design decisions, and the measurements behind them

**Frequency decides which keys matter; permissive sources decide the answer.**
Counts are aggregate and non-identifying, so they may come from restricted
corpora. Romanizations may only come from redistributable sources. That single
split resolves the licensing and the privacy question together.

**The trunk is tiny.** Over 4M rows of the Punjab electoral roll, the top 23
`elector_name` types cover 50% of tokens and the top 16,466 cover 99%, out of
70,753 types. You never have to boil the ocean.

**Lookup keys never fold on a Unicode heuristic.** An earlier design folded
vowel length, nasalization, gemination and the AA matra as "scribal noise".
Ablating each component over the 3,833 roll surfaces with ≥100 occurrences
showed every one merging more distinct words than duplicate spellings:

| component | merges | benign | harmful |
|---|---|---|---|
| vowel length | 73 | 43 | 30 |
| diphthong | 62 | 25 | 37 |
| drop nasal | 83 | 6 | 77 |
| drop addak | 73 | 13 | 60 |
| drop nukta | 130 | 6 | 124 |
| drop AA matra | 198 | 23 | 175 |

`ਬੁਟਾ`/*buta* and `ਬੂਟਾ`/*boota* are different names. Folding survives only as
`alias_candidate_key`, which *proposes* merges for evidence to confirm.

**Voting is over trust groups, not source names.** `data/punjabi.csv.gz` was
extracted from the Punjab roll, and the shipped Punjabi model was trained on
that corpus — one opinion in three hats. `sources.py` puts them in one
`trust_group` so they cannot manufacture agreement.

**A single source cannot confer authority.** An unscoped Wikidata pull asserted
`ਮਸੀਹ`→`christ` (130,862 roll occurrences), `ਪਾਲ`→`paul` (49,929) and
`ਚੱਕ`→`chuck` (45,518). Every query is bound to `wd:Q668`, and `high`
confidence additionally requires two independent trust groups.

**Machines rank but do not authorize.** Sources with `human_attested=False`
contribute to scores and break ties; a candidate they alone support never
exceeds `low`. This is what stops "the gazetteer beats the LLM" from meaning
"the LLM agrees with itself".

**Attestation multiplicity is part of the weight.** Collapsing duplicates and
keeping max weight let one accidental alignment count as much as thousands of
consistent ones; every margin flattened and almost nothing reached `high`.
Weighting by within-source attestation share raised certified coverage of Hindi
token mass from 16.9% to 46.8%.

**Length plausibility is measured, not guessed.** Taking the 2,106 Hindi pairs
where ≥2 independent sources already agree, the ratio of Latin characters to
aksharas runs p0.5 = 0.50 to p99.5 = 3.00. That band drops 2.8% of harvested
rows while rejecting only 6 of the 2,106 agreed pairs — and what it removes is
misalignment (`भारत`→`deindustrialization`, `के`→`administrative`).

## Licensing

The corpus is published **CC-BY-4.0**, which is what CC0 (Wikidata) + CC-BY-4.0
(GeoNames) + repo-owned inputs permit. `sources.py` marks everything else
`redistributable=False` and `build.py` reads only bundled sources, so the
licence claim is enforced in code rather than asserted in prose.

Excluded and why: OSM (ODbL share-alike), Dakshina and Wikipedia interwiki
(CC-BY-SA), `data/hindi.csv.gz` and `data/iit/` (IIT Bombay mined pairs are
CC-BY-NC), the Punjab roll and `data/punjabi.csv.gz` (restricted, and not
independent).

Aksharantar was listed here as CC-BY-NC. **That was wrong** — the dataset card
licenses manual data CC-BY and mined/existing data CC0. It stays out for now
because `data/README.md` records that it contains Dakshina-sourced rows, and
Dakshina is the only held-out scoreboard below; that reason needs confirming
before 1.8M permissive pairs are written off. See the limitations section.

Dakshina being licence-excluded and being the natural held-out evaluation set is
the same decision, which is convenient: it cannot leak into what it evaluates.

## Pipeline

```bash
# Stage 1 -- frequency: which keys matter (needs the train group for pyarrow)
uv run --group train python -m gazetteer.mine_frequency --lang hindi
uv run --group train python -m gazetteer.mine_frequency --lang punjabi

# Stage 2 -- harvest: what each source claims
uv run python -m gazetteer.harvest_wikidata --lang hindi
uv run python -m gazetteer.harvest_geonames --lang hindi   # downloads the India dumps once
uv run python -m gazetteer.harvest_corpus  --lang hindi

# Stage 3 -- adjudicate and report
uv run python -m gazetteer.build --lang hindi

# Stage 4 -- is it right? Score against Dakshina, which is not one of its sources
uv run --group train python -m gazetteer.conviction --lang hindi
```

Outputs land under `gazetteer/build/` (git-ignored): `freq/` the key lists,
`src/` per-source claims, `corpus/` the corpus JSONL plus its conviction report.

## Modules

| module | role |
|---|---|
| `script.py` | Indic detection, per-language script gate, clean-token rule |
| `align.py` | positional label → token-pair decomposition |
| `sources.py` | licence posture, trust groups, authority priors |
| `records.py` | validated row schema; attestation-weighted `aggregate` |
| `frequency.py` | Zipf mining, coverage cutpoints |
| `plausibility.py` | measured length band |
| `phonetic.py` | rule-based romanization check; rejects translations |
| `adjudicate.py` | noisy-OR over trust groups, confidence tiers |
| `build.py` | assembly and the coverage/agreement/contamination report |
| `conviction.py` | held-out accuracy against Dakshina |

Key normalization lives in `indicate/normalize.py`, not here, so corpus keys and
lookup keys cannot drift.

## How good is it, actually

Scored against the Dakshina romanization lexicon — which `conviction.py` asserts
is not one of the corpus's sources before reporting anything — on the 541 Hindi
keys the two share:

| | keys | exact match |
|---|---|---|
| corpus, top-ranked candidate | 541 | 79.1% |
| **shipped seq2seq model, same words** | 541 | **82.8%** |
| corpus, `high` tier only | 109 | 84.4% |
| corpus, `medium` tier only | 432 | 77.8% |
| corpus, contested keys only | 146 | 70.5% |

**Read that honestly: as a general table the corpus is currently 3 points behind
the model already in the wheel.** The confidence tiers do separate, and the
`high` tier does beat the model — but it is 20% of the overlap. So the reason to
keep building this is not accuracy. It is that Wikidata and GeoNames are CC0/
CC-BY and the model's training corpora are not, which makes this the only
redistributable route to a general table. Accuracy is the thing that still has
to be earned, and the number above is the scoreboard.

What has been tried, and what each move actually bought:

| change | accuracy | `high` keys | token-mass coverage |
|---|---|---|---|
| baseline | 74.7% | 1,751 | 91.5% |
| + per-language script gate | 74.7% | 1,751 | 91.5% |
| + phonetic filter (`phonetic.py`) | **79.5%** | 1,737 | 88.7% |
| + GeoNames as a second CC-BY group | 79.1% | **2,785** | 88.7% |

The phonetic filter is worth 4.8 points because Wikidata's Latin label for an
entity is frequently the English *name* rather than a romanization — `उड़ान` →
`flight`, `क्षेत्रों` → `of`, `कब्र` → `forbes`. No length or plausibility check can
see those; comparison against a rule-based ISO/IAST romanization can.

GeoNames did the job it was added for and no more. Corroboration rose 60% —
`high` requires two independent trust groups, and until now place names had only
Wikidata — but accuracy did not move (79.5% → 79.1%, about two words out of 541).
The reason is supply: **all of India has 1,440 Hindi and 265 Punjabi alternate
names in the GeoNames dump.** The pairs it does yield are good (`महू` → `mhow`,
`वइटिला` → `vyttila`), there are simply not many. So the earlier guess that
GeoNames was the biggest lever left was wrong: it corroborates, it does not
cover.

## Known limitations

- **Hindi frequency is weak.** The only Devanagari frequency source in-repo is
  `data/affidavits.csv`, whose head is dominated by single-letter initials
  (के, एस, ए). That is a fact about affidavit formatting, not about Hindi. The
  Punjab roll is a far better frequency oracle and has no Devanagari analogue
  here yet.
- **Corroboration is still thin, and GeoNames could not fix it.** 88.7% of Hindi
  token mass has at least one candidate; only 44.4% is backed by two independent
  trust groups. GeoNames is now wired up and raised `high` keys by 60%, but with
  1,440 Hindi names in the entire India dump it cannot do more. A third
  permissive Indic-script source is the open problem — which is the same format
  gap this corpus was built to fill.
- **The obvious candidate may be Aksharantar, wrongly written off.** It is
  CC-BY/CC0, not CC-BY-NC as this file previously claimed, and it is 1.3M Hindi
  and 515k Punjabi pairs already sitting in `data/aksharantar/`. The blocker is
  not licensing but evaluation: it carries Dakshina-sourced rows, so admitting
  it contaminates the only held-out set the numbers above rest on. Using it
  means finding a second scoreboard first.
- **Some remaining "errors" are conventions, not mistakes.** The corpus says
  `ghazipur`, `ganapathy`, `gayathri`; Dakshina's crowd spellings say `gajipur`,
  `ganapati`, `gayatri`. Both are defensible, and the 79.1% counts them as
  wrong, so it is a floor rather than an estimate.
