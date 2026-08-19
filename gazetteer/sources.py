"""Registry of romanization sources, with licensing and independence metadata.

Two properties of a source matter more than its size.

**Redistributability.** The published corpus is licensed CC-BY-4.0, which is
exactly what CC0 and CC-BY inputs permit. Share-alike inputs (OSM's ODbL,
Dakshina and Wikipedia's CC-BY-SA) would force the whole derived database to
inherit their terms. Those sources stay registered -- they are useful for
evaluation and analysis -- but ``redistributable=False`` keeps them out of the
bundle, and ``bundled_sources`` is the single place that decision is enforced.

Aksharantar was previously excluded here as CC-BY-NC. **That was wrong**: the
dataset card licenses manually collected data CC-BY and the mined and existing
portions CC0, and the Hugging Face metadata tag is the generic ``cc``. It is
still not bundled, but for a different reason that has yet to be confirmed --
``data/README.md`` records that Aksharantar contains Dakshina-sourced rows, and
Dakshina is the corpus's only held-out scoreboard. Resolve before adding it; a
wrong reason recorded as a right one is how 1.8M permissive pairs went unused
while this file called source supply the open problem.

**Independence.** Cross-source voting is only meaningful between sources that
could disagree. ``data/punjabi.csv.gz`` was extracted from the Punjab electoral
roll, and the shipped Punjabi model was in turn trained on it, so roll, corpus
and model are one opinion wearing three hats. ``trust_group`` collapses them;
:func:`independent_groups` is what the adjudicator counts instead of raw source
names. Without it a single GPT-4o annotation pass would outvote every genuinely
independent source.

``human_attested`` is the companion guard: LLM-derived sources may rank and
break ties, but a candidate cannot reach high confidence on their word alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Metadata describing one romanization source.

    Attributes:
        name: Registry key, also the provenance tag written into the corpus.
        licence: SPDX-style identifier, or ``"repo-own"`` for data this project
            collected itself.
        redistributable: Whether entries may enter the published corpus.
        trust_group: Sources sharing a group are one opinion, not several.
        human_attested: False for machine-generated romanizations.
        authority: Prior weight in ``[0, 1]`` used by the adjudicator.
        note: Why the source is treated the way it is.
    """

    name: str
    licence: str
    redistributable: bool
    trust_group: str
    human_attested: bool
    authority: float
    note: str = ""


SOURCES: dict[str, SourceSpec] = {
    spec.name: spec
    for spec in (
        # --- bundled: CC0 / CC-BY / repo-owned -------------------------------
        SourceSpec(
            name="wikidata",
            licence="CC0-1.0",
            redistributable=True,
            trust_group="wikidata",
            human_attested=True,
            authority=0.85,
            note="India-scoped labels; must filter on P17/P27=Q668 or it emits "
            "translations and foreign homographs (ਮਸੀਹ->christ, ਪਾਲ->paul).",
        ),
        SourceSpec(
            name="geonames",
            licence="CC-BY-4.0",
            redistributable=True,
            trust_group="geonames",
            human_attested=True,
            authority=0.80,
            note="alternateNames keyed to geonameid; Indic fill rate unverified.",
        ),
        SourceSpec(
            name="affidavits",
            licence="repo-own",
            redistributable=True,
            trust_group="myneta",
            human_attested=True,
            authority=0.70,
            note="data/affidavits.csv; candidate-declared bilingual name pairs.",
        ),
        SourceSpec(
            name="cricinfo",
            licence="repo-own",
            redistributable=True,
            trust_group="cricinfo",
            human_attested=True,
            authority=0.70,
            note="data/players_with_hindi_names.json; editorially maintained.",
        ),
        # --- registered but never bundled ------------------------------------
        SourceSpec(
            name="punjab_roll",
            licence="restricted",
            redistributable=False,
            trust_group="punjab_roll",
            human_attested=False,
            authority=0.40,
            note="GPT-4o annotations over an IRB-restricted deposit. Used for "
            "frequency ranking (aggregate counts only) and as a low-authority "
            "tiebreak; never as evidence of authority on its own.",
        ),
        SourceSpec(
            name="punjabi_corpus",
            licence="restricted",
            redistributable=False,
            trust_group="punjab_roll",
            human_attested=False,
            authority=0.40,
            note="data/punjabi.csv.gz, extracted from punjab_roll; the shipped "
            "Punjabi model was trained on it. Same trust group by construction.",
        ),
        SourceSpec(
            name="hindi_corpus",
            licence="CC-BY-NC-4.0",
            redistributable=False,
            trust_group="iitb",
            human_attested=True,
            authority=0.50,
            note="data/hindi.csv.gz blends repo-own scrapes with IIT Bombay "
            "mined pairs (NC). Harvest affidavits/cricinfo directly instead.",
        ),
        SourceSpec(
            name="iitb",
            licence="CC-BY-NC-4.0",
            redistributable=False,
            trust_group="iitb",
            human_attested=True,
            authority=0.50,
            note="data/iit/en-hi.mined-pairs.",
        ),
        SourceSpec(
            name="dakshina",
            licence="CC-BY-SA-4.0",
            redistributable=False,
            trust_group="dakshina",
            human_attested=True,
            authority=0.90,
            note="Held out as the evaluation set. Excluding it from the bundle "
            "on licence grounds and holding it out for leakage control are the "
            "same decision.",
        ),
        SourceSpec(
            name="aksharantar",
            licence="CC-BY-NC-4.0",
            redistributable=False,
            trust_group="aksharantar",
            human_attested=True,
            authority=0.60,
            note="Licence ambiguous: HF metadata says NC, card body says "
            "CC-BY/CC0 by subset. Excluded until resolved.",
        ),
        SourceSpec(
            name="osm",
            licence="ODbL-1.0",
            redistributable=False,
            trust_group="osm",
            human_attested=True,
            authority=0.75,
            note="In India `name` is already Latin, so the pair is "
            "(name:xx, name). Share-alike; user-buildable layer only.",
        ),
        SourceSpec(
            name="wikipedia_interwiki",
            licence="CC-BY-SA-4.0",
            redistributable=False,
            trust_group="wikipedia",
            human_attested=True,
            authority=0.65,
            note="The data/railway_stations and data/wikipedia_interwiki "
            "scrapers. Wikipedia content is share-alike.",
        ),
    )
}


def bundled_sources() -> list[SourceSpec]:
    """Return the sources whose entries may enter the published corpus.

    Returns:
        Every registered source with ``redistributable=True``, ordered by
        descending authority.
    """
    return sorted(
        (s for s in SOURCES.values() if s.redistributable),
        key=lambda s: (-s.authority, s.name),
    )


def independent_groups(names: Iterable[str]) -> set[str]:
    """Collapse source names to the set of genuinely independent opinions.

    Args:
        names: Source names, possibly including derived copies of one another.

    Returns:
        The distinct trust groups they belong to. Unknown names are ignored
        rather than counted, so a typo cannot manufacture independence.
    """
    return {SOURCES[n].trust_group for n in names if n in SOURCES}


def iter_bundled_names() -> Iterator[str]:
    """Yield the names of bundled sources, highest authority first.

    Yields:
        Source names.
    """
    for spec in bundled_sources():
        yield spec.name
