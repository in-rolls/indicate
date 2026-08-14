"""Tests for the source registry: licensing posture and independence groups."""

import unittest
from dataclasses import FrozenInstanceError

from gazetteer.sources import (
    SOURCES,
    SourceSpec,
    bundled_sources,
    independent_groups,
)


class TestRegistryIntegrity(unittest.TestCase):
    def test_every_source_is_keyed_by_its_own_name(self):
        for name, spec in SOURCES.items():
            self.assertEqual(name, spec.name)

    def test_authority_is_a_probability(self):
        for spec in SOURCES.values():
            self.assertGreaterEqual(spec.authority, 0.0, spec.name)
            self.assertLessEqual(spec.authority, 1.0, spec.name)

    def test_every_source_declares_a_licence_and_trust_group(self):
        for spec in SOURCES.values():
            self.assertTrue(spec.licence, spec.name)
            self.assertTrue(spec.trust_group, spec.name)


class TestLicensingPosture(unittest.TestCase):
    """Permissive-only bundle: CC0 + CC-BY + repo-owned, nothing else."""

    def test_share_alike_and_non_commercial_sources_are_not_bundled(self):
        bundled = {s.name for s in bundled_sources()}
        for name in ("osm", "dakshina", "aksharantar", "wikipedia_interwiki", "iitb"):
            self.assertIn(name, SOURCES, f"{name} should be registered")
            self.assertNotIn(name, bundled, f"{name} is not redistributable")

    def test_permissive_sources_are_bundled(self):
        bundled = {s.name for s in bundled_sources()}
        for name in ("wikidata", "geonames", "affidavits", "cricinfo"):
            self.assertIn(name, bundled)

    def test_no_bundled_source_carries_a_viral_or_nc_licence(self):
        for spec in bundled_sources():
            self.assertNotIn("SA", spec.licence, spec.name)
            self.assertNotIn("NC", spec.licence, spec.name)

    def test_hindi_corpus_is_excluded_because_it_mixes_in_iitb_nc_pairs(self):
        # data/hindi.csv.gz blends repo-own scrapes with IIT Bombay mined pairs
        # (CC-BY-NC), so the harvesters read affidavits/cricinfo directly instead.
        self.assertNotIn("hindi_corpus", {s.name for s in bundled_sources()})


class TestCircularityGuard(unittest.TestCase):
    """Voting must not treat derived copies of one source as independent."""

    def test_punjabi_corpus_shares_a_trust_group_with_the_roll_it_came_from(self):
        # data/punjabi.csv.gz was extracted from the Punjab roll, and the shipped
        # model was trained on it. Counting them separately would let a single
        # GPT-4o annotation outvote genuinely independent sources.
        self.assertEqual(
            SOURCES["punjabi_corpus"].trust_group, SOURCES["punjab_roll"].trust_group
        )

    def test_llm_derived_sources_are_not_human_attested(self):
        for name in ("punjab_roll", "punjabi_corpus"):
            self.assertFalse(SOURCES[name].human_attested, name)

    def test_reference_sources_are_human_attested(self):
        for name in ("wikidata", "geonames"):
            self.assertTrue(SOURCES[name].human_attested, name)

    def test_independent_groups_collapses_derived_sources(self):
        self.assertEqual(
            independent_groups(["punjab_roll", "punjabi_corpus"]),
            {SOURCES["punjab_roll"].trust_group},
        )

    def test_independent_groups_counts_distinct_sources_separately(self):
        self.assertEqual(len(independent_groups(["wikidata", "geonames"])), 2)

    def test_independent_groups_ignores_unknown_names(self):
        self.assertEqual(independent_groups(["nope"]), set())


class TestSourceSpec(unittest.TestCase):
    def test_is_immutable(self):
        spec = SOURCES["wikidata"]
        with self.assertRaises(FrozenInstanceError):
            spec.authority = 0.1  # type: ignore[misc]

    def test_can_describe_an_unregistered_source(self):
        spec = SourceSpec(
            name="local",
            licence="CC0-1.0",
            redistributable=True,
            trust_group="local",
            human_attested=True,
            authority=0.5,
        )
        self.assertEqual(spec.name, "local")


if __name__ == "__main__":
    unittest.main()
