"""The candidate-row schema shared by every harvester and the adjudicator.

A harvester's whole job is to turn some source into rows of this shape. Keeping
the schema in one place, with validation at construction, means provenance is
always checkable: a row cannot claim a source that is not registered, and a
Latin side cannot slip through un-normalized and split one candidate into two
(``Rāj`` voting separately from ``raj`` is exactly the failure this prevents).
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gazetteer.script import is_clean_native_token
from gazetteer.sources import SOURCES
from indicate.normalize import gaz_key, latin_form

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

FIELDNAMES = ("native", "latin", "source", "entity_type", "weight", "ref")


@dataclass(frozen=True, slots=True)
class CandidateRow:
    """One source's claim that ``native`` romanizes to ``latin``.

    Attributes:
        native: The token in its Indic-script surface form.
        latin: The romanization, already in :func:`~indicate.normalize.latin_form`.
        source: A key of :data:`~gazetteer.sources.SOURCES`.
        entity_type: ``"person"``, ``"geo"``, ``"org"`` or ``"vocab"``.
        weight: Within-source confidence in ``[0, 1]``, e.g. scaled attestation
            count. Cross-source authority comes from the registry, not here.
        ref: Source-local identifier, such as a Wikidata QID or geonameid.
    """

    native: str
    latin: str
    source: str
    entity_type: str
    weight: float
    ref: str = ""

    def __post_init__(self) -> None:
        """Validate the row.

        Raises:
            ValueError: If the source is unregistered, either side is empty,
                the native side is not a clean Indic token, the Latin side is
                not normalized, or the weight is out of range.
        """
        if self.source not in SOURCES:
            raise ValueError(f"unregistered source: {self.source!r}")
        if not self.native.strip():
            raise ValueError("native side is empty")
        if not is_clean_native_token(self.native):
            # Latin-on-both-sides pairs teach nothing about romanization, and
            # mixed-script hybrids are tokenization artifacts, not words.
            raise ValueError(f"native side is not a clean Indic token: {self.native!r}")
        if not self.latin.strip():
            raise ValueError("latin side is empty")
        if self.latin != latin_form(self.latin):
            raise ValueError(
                f"latin side is not normalized: {self.latin!r} "
                f"(expected {latin_form(self.latin)!r})"
            )
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight out of range: {self.weight!r}")


def write_rows(path: Path, rows: Iterable[CandidateRow]) -> int:
    """Write candidate rows as TSV, creating parent directories as needed.

    Args:
        path: Destination file.
        rows: Rows to write.

    Returns:
        The number of rows written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDNAMES, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "native": row.native,
                    "latin": row.latin,
                    "source": row.source,
                    "entity_type": row.entity_type,
                    "weight": f"{row.weight:.4f}",
                    "ref": row.ref,
                }
            )
            written += 1
    return written


def read_rows(path: Path) -> list[CandidateRow]:
    """Read candidate rows from a TSV written by :func:`write_rows`.

    Args:
        path: Source file.

    Returns:
        The rows, or ``[]`` if the file does not exist.
    """
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [
            CandidateRow(
                native=record["native"],
                latin=record["latin"],
                source=record["source"],
                entity_type=record["entity_type"],
                weight=float(record["weight"]),
                ref=record.get("ref") or "",
            )
            for record in reader
        ]


def aggregate(rows: Sequence[CandidateRow]) -> list[CandidateRow]:
    """Collapse repeated claims into one row per pair, weighted by attestation.

    A source's opinion about a key is distributed across the romanizations it
    proposes, in proportion to how often it proposes each. Without this a single
    accidental alignment (one stray "leo" for a common surname) would otherwise
    carry the same weight as thousands of consistent ones ("singh"), which
    flattens every margin and stops obvious entries reaching high confidence.

    Weights are normalized per (key, source), so a prolific source cannot
    outvote a careful one -- cross-source authority is applied separately by the
    adjudicator.

    Args:
        rows: Raw rows from a single harvest.

    Returns:
        One row per (key, source, latin), weight equal to that pair's share of
        the source's attestations for the key, in a deterministic order.
    """
    counts: Counter[tuple[str, str, str]] = Counter()
    exemplar: dict[tuple[str, str, str], CandidateRow] = {}
    for row in rows:
        ident = (gaz_key(row.native), row.source, row.latin)
        counts[ident] += 1
        exemplar.setdefault(ident, row)

    totals: Counter[tuple[str, str]] = Counter()
    for (key, source, _), count in counts.items():
        totals[(key, source)] += count

    out: list[CandidateRow] = []
    for ident in sorted(counts):
        key, source, _ = ident
        row = exemplar[ident]
        out.append(
            CandidateRow(
                native=row.native,
                latin=row.latin,
                source=row.source,
                entity_type=row.entity_type,
                weight=counts[ident] / totals[(key, source)],
                ref=row.ref,
            )
        )
    return out
