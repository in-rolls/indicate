"""Token frequency mining: deciding which keys are worth looking up.

The gazetteer strategy rests on one empirical fact: Indic name and place tokens
are severely Zipf-distributed, so a small table captures most real-world token
mass. Measured over 4M rows of the Punjab electoral roll, the top 23
``elector_name`` types cover 50% of tokens and the top 16,466 cover 99%, out of
70,753 types. Place fields are more concentrated still.

Frequency is mined separately from romanization on purpose. Counts are aggregate
and non-identifying, so they can come from restricted corpora that must never
contribute a romanization; the answer side is restricted to redistributable
sources. That split is what keeps the published corpus clean.

Entity hints are accumulated as a multiset rather than a label. ``ਸਿੰਘ`` is
overwhelmingly a surname but also appears in ``main_town``, and pretending
otherwise would let a place romanization capture every surname occurrence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from gazetteer.script import has_indic_content, strip_edge_noise
from indicate.normalize import gaz_key

#: Coverage thresholds reported by default; the trunk size is chosen from these
#: rather than from a magic entry count.
DEFAULT_MARKS = (0.5, 0.8, 0.9, 0.95, 0.99, 0.995)


def iter_tokens(text: str | None) -> Iterator[str]:
    """Yield lookup-worthy Indic tokens from a raw corpus field.

    Args:
        text: A field value such as an elector name or a town name.

    Yields:
        Whitespace-separated tokens with edge punctuation and administrative
        digit prefixes removed. Tokens containing no Indic letter are skipped:
        bare codes and already-Latin text carry no transliteration signal.
    """
    if not text:
        return
    for raw in text.split():
        token = strip_edge_noise(raw)
        if token and has_indic_content(token):
            yield token


@dataclass(frozen=True, slots=True)
class FrequencyRow:
    """One key's frequency evidence.

    Attributes:
        key: Canonical lookup key.
        count: Total occurrences across every surface form.
        surfaces: Observed spellings mapped to their counts.
        entities: Field classes the key was seen in, mapped to their counts.
    """

    key: str
    count: int
    surfaces: Mapping[str, int]
    entities: Mapping[str, int]


@dataclass(slots=True)
class FrequencyTable:
    """Accumulates token counts, surface spellings and entity hints by key."""

    _counts: Counter[str] = field(default_factory=Counter)
    _surfaces: defaultdict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    _entities: defaultdict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def add(self, surface: str, entity: str, n: int = 1) -> None:
        """Record ``n`` occurrences of ``surface`` seen as an ``entity``.

        Args:
            surface: The token as it appeared in the corpus.
            entity: Field class it came from, e.g. ``"person"`` or ``"geo"``.
            n: Occurrence count to add.
        """
        key = gaz_key(surface)
        if not key:
            return
        self._counts[key] += n
        self._surfaces[key][surface] += n
        self._entities[key][entity] += n

    def add_all(self, surfaces: Iterable[str], entity: str) -> None:
        """Record one occurrence of each token in ``surfaces``.

        Args:
            surfaces: Tokens as they appeared in the corpus.
            entity: Field class they came from.
        """
        for surface in surfaces:
            self.add(surface, entity)

    def count(self, surface: str) -> int:
        """Return the recorded count for ``surface``'s key.

        Args:
            surface: Any spelling of the token.

        Returns:
            Total occurrences, or ``0`` if the key was never seen.
        """
        return self._counts[gaz_key(surface)]

    @property
    def total_tokens(self) -> int:
        """Total occurrences recorded across all keys."""
        return sum(self._counts.values())

    def rows(self) -> list[FrequencyRow]:
        """Return every key's evidence, most frequent first.

        Returns:
            Rows sorted by descending count then ascending key, so repeated
            builds are byte-identical.
        """
        return [
            FrequencyRow(
                key=key,
                count=count,
                surfaces=dict(self._surfaces[key].most_common()),
                entities=dict(self._entities[key].most_common()),
            )
            for key, count in sorted(
                self._counts.items(), key=lambda kv: (-kv[1], kv[0])
            )
        ]

    def coverage(self, marks: Sequence[float] = DEFAULT_MARKS) -> dict[float, int]:
        """Report how many keys are needed to reach each share of token mass.

        Args:
            marks: Cumulative token-mass thresholds.

        Returns:
            Each threshold mapped to the number of keys required.
        """
        return coverage_cutpoints(list(self._counts.values()), marks=marks)


def coverage_cutpoints(
    counts: Iterable[int], marks: Sequence[float] = DEFAULT_MARKS
) -> dict[float, int]:
    """Return how many of the largest counts are needed to reach each threshold.

    Args:
        counts: Per-type occurrence counts, in any order.
        marks: Cumulative shares of total mass to report.

    Returns:
        Each threshold mapped to the smallest number of types whose combined
        count reaches it. All zero when ``counts`` is empty.
    """
    ordered = sorted(counts, reverse=True)
    total = sum(ordered)
    if not total:
        return dict.fromkeys(marks, 0)

    result: dict[float, int] = {}
    pending = sorted(marks)
    cumulative = 0
    for index, value in enumerate(ordered, start=1):
        cumulative += value
        while pending and cumulative / total >= pending[0]:
            result[pending.pop(0)] = index
        if not pending:
            break
    for mark in pending:  # floating-point shortfall on the last mark
        result[mark] = len(ordered)
    return {mark: result[mark] for mark in marks}
