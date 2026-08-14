"""Build pipeline for the open Indic romanization gazetteer.

This package is the corpus builder, not part of the shipped library: it mines
token frequency from real corpora, harvests candidate romanizations from
independent sources, and adjudicates them into a ranked, provenance-tagged
table. It is excluded from the wheel and the sdist.

It imports :mod:`indicate.normalize` rather than reimplementing key
normalization, so corpus keys and lookup keys cannot drift apart.
"""
