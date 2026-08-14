"""Cross-source adjudication: turning raw claims into a ranked, tiered corpus.

The corpus does not assert one canonical romanization. It asserts a *ranked*
list with provenance and a confidence tier, because "official transliteration"
has no single registry behind it -- Survey of India's 1.4 million ground-verified
names are published as map labels, not as data, and everything downloadable is
someone's attestation rather than a register.

Three rules make the ranking trustworthy.

**Independence, not headcount.** Scores combine by noisy-OR over *trust groups*
(:func:`~gazetteer.sources.independent_groups`), never over source names. The
Punjab roll, the Punjabi corpus extracted from it, and the model trained on that
corpus are one opinion; counting them three times would let a single GPT-4o
annotation pass outvote every independent source.

**A single source cannot confer authority.** An unscoped Wikidata pull asserted
``ਮਸੀਹ``->``christ`` on a token with 130,862 roll occurrences. Reputation is not
corroboration, so ``high`` requires two independent groups.

**Machines can rank but not authorize.** Sources with ``human_attested=False``
contribute to scores and break ties, but a candidate supported only by them
never exceeds ``low``. That is what keeps "the gazetteer beats the LLM" from
meaning "the LLM agrees with itself".
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from gazetteer.phonetic import is_phonetic_match
from gazetteer.plausibility import is_plausible_pair
from gazetteer.records import CandidateRow
from gazetteer.sources import SOURCES
from indicate.normalize import gaz_key

#: Minimum gap between the top two candidates for the winner to be authoritative.
#: Two reputable sources that disagree is a disagreement, not authority.
MIN_MARGIN = 0.15

HIGH = "high"
MEDIUM = "medium"
LOW = "low"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One romanization and the evidence behind it.

    Attributes:
        latin: The romanization.
        score: Noisy-OR combination over independent trust groups, in ``[0, 1]``.
        sources: Every source that asserted it, sorted.
        groups: The distinct trust groups those sources belong to, sorted.
    """

    latin: str
    score: float
    sources: tuple[str, ...]
    groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Adjudication:
    """The corpus entry for one key.

    Attributes:
        key: Canonical lookup key.
        candidates: Ranked romanizations, best first.
        confidence: ``"high"``, ``"medium"`` or ``"low"``.
        margin: Score gap between the top two candidates.
        entity: Entity types the key was attested as, with counts.
    """

    key: str
    candidates: tuple[Candidate, ...]
    confidence: str
    margin: float
    entity: Mapping[str, int]


def _group_evidence(rows: Sequence[CandidateRow]) -> dict[str, float]:
    """Return the best evidence strength each trust group offers."""
    best: dict[str, float] = {}
    for row in rows:
        spec = SOURCES[row.source]
        strength = spec.authority * row.weight
        if strength > best.get(spec.trust_group, 0.0):
            best[spec.trust_group] = strength
    return best


def _noisy_or(strengths: Sequence[float]) -> float:
    """Combine independent evidence: the chance at least one group is right."""
    complement = 1.0
    for strength in strengths:
        complement *= 1.0 - strength
    return 1.0 - complement


def adjudicate_key(
    rows: Sequence[CandidateRow], *, min_margin: float = MIN_MARGIN
) -> Adjudication:
    """Adjudicate every claim about a single key.

    Args:
        rows: Candidate rows that all share one canonical key.
        min_margin: Score gap the winner must clear to be authoritative.

    Returns:
        The ranked, tiered corpus entry.

    Raises:
        ValueError: If ``rows`` is empty or spans more than one key.
    """
    if not rows:
        raise ValueError("no rows to adjudicate")
    keys = {gaz_key(row.native) for row in rows}
    if len(keys) != 1:
        raise ValueError(f"rows span multiple keys: {sorted(keys)}")
    key = keys.pop()

    by_latin: defaultdict[str, list[CandidateRow]] = defaultdict(list)
    entity: Counter[str] = Counter()
    for row in rows:
        by_latin[row.latin].append(row)
        entity[row.entity_type] += 1

    candidates: list[Candidate] = []
    for latin, latin_rows in by_latin.items():
        evidence = _group_evidence(latin_rows)
        candidates.append(
            Candidate(
                latin=latin,
                score=_noisy_or(list(evidence.values())),
                sources=tuple(sorted({r.source for r in latin_rows})),
                groups=tuple(sorted(evidence)),
            )
        )
    # Sort by score, then by latin so repeated builds are byte-identical.
    candidates.sort(key=lambda c: (-c.score, c.latin))

    winner = candidates[0]
    runner_up = candidates[1].score if len(candidates) > 1 else 0.0
    margin = winner.score - runner_up

    return Adjudication(
        key=key,
        candidates=tuple(candidates),
        confidence=_confidence(winner, by_latin[winner.latin], margin, min_margin),
        margin=margin,
        entity=dict(entity.most_common()),
    )


def _confidence(
    winner: Candidate,
    winner_rows: Sequence[CandidateRow],
    margin: float,
    min_margin: float,
) -> str:
    """Assign a confidence tier to the winning candidate."""
    human_backed = any(SOURCES[row.source].human_attested for row in winner_rows)
    if not human_backed:
        # Machine-generated evidence can rank, but never authorize.
        return LOW
    if len(winner.groups) >= 2 and margin >= min_margin:
        return HIGH
    return MEDIUM


def adjudicate_all(
    rows: Sequence[CandidateRow],
    *,
    min_margin: float = MIN_MARGIN,
    lang: str | None = None,
) -> list[Adjudication]:
    """Adjudicate a whole harvest, grouping rows by canonical key.

    Rows failing :func:`~gazetteer.plausibility.is_plausible_pair` are discarded
    first; a key left with no plausible pair produces no entry. When ``lang`` is
    given, rows failing :func:`~gazetteer.phonetic.is_phonetic_match` go too --
    that filter needs the language to know which script it is reading.

    Args:
        rows: Candidate rows from every source.
        min_margin: Score gap the winner must clear to be authoritative.
        lang: Corpus language name; ``None`` skips the phonetic filter.

    Returns:
        One entry per surviving key, ordered by key so builds are reproducible.
    """
    grouped: defaultdict[str, list[CandidateRow]] = defaultdict(list)
    for row in rows:
        # Implausible pairs are alignment noise, not competing candidates.
        # Admitting them would manufacture disagreement and depress margins.
        if not is_plausible_pair(row.native, row.latin):
            continue
        # A translation has an ordinary length and the wrong relationship, so
        # length cannot see it; resemblance to a rule-based romanization can.
        if lang is not None and not is_phonetic_match(row.native, row.latin, lang):
            continue
        grouped[gaz_key(row.native)].append(row)
    return [
        adjudicate_key(grouped[key], min_margin=min_margin) for key in sorted(grouped)
    ]
