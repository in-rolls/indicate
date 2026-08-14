"""Fast word-level lookup: answer from a table, decode only the tail.

On the Punjab electoral roll a table built from ``data/punjabi.csv.gz`` answers
**99.1% of token mass**, so the decoder handles 0.9% of the work. Measured back
to back on one machine (``training/bench_lookup.py``, warm filesystem cache;
absolute values move with machine load, the ratios do not):

=============================  ============  ============  =======
measurement                    with lookup   model only    ratio
=============================  ============  ============  =======
end-to-end on roll names       10,937 tok/s  258 tok/s     42x
cold start, all-hit input      0.10 s        0.44 s        4.4x
raw table read (zipf, cached)  16.9M/s       --            --
=============================  ============  ============  =======

Cold start is the smaller ratio but the one users feel per invocation, and it
is only 4.4x because a fully-hit input **never imports torch**. That property is
load-bearing, not incidental: it is why the torch-importing modules are imported
inside :meth:`indicate.engine.ModelBackend.resolve` rather than at module scope,
and ``tests/test_lookup_bench.py`` fails if anything moves them back.

Accuracy does not pay for that speed; it improves. On the Dakshina test words a
table happens to contain, answering from the table and falling through to the
model on the rest beats either alone -- Punjabi 77.6% against the model's 77.0%,
Hindi 78.8% against 76.2%. The gap comes from ``training/build_lookup.py``
refusing to answer where the corpus has no majority, so ambiguity reaches the
model instead of being resolved by a coin toss.

There is a reason the two are complementary rather than redundant.
``training/build_v2.py`` withholds from training every source word appearing in
the evaluation sets, which removed ``ਕੌਰ``, ``ਕੁਮਾਰ``, ``ਦਾਸ``, ``ਬਟਾਲਾ`` and
``ਗੁਰਦਾਸਪੁਰ`` -- the most frequent tokens in the corpus. The model is weakest
precisely where the table is strongest, which is why it scrambles ``ਨਗਰ`` into
``ganag``.

**A table carries a romanization convention.** The roll table speaks the
long-vowel style of its GPT-4o annotation (``azaadi``, ``apaar``, ``ambee``)
where Dakshina prefers ``azadi``, ``apar``, ``ambi``. Neither is wrong, but a
string that mixes table hits with model misses could mix styles, so the
``convention`` header records which one a table speaks.

That worry turns out to be small, and ``training/seam_check.py`` is what
measured it rather than assuming. On roll strings containing both a hit and a
miss, the table and the model already produce the *same* string for 93.0% of hit
words. On the rest the style gap between the hit half and the miss half is
+0.0095 doubled vowels per character wider than in the all-model rendering of
the same input -- roughly one extra long vowel per 100 characters. The direction
is not even consistent (``ਕੁਮਾਰੀ``: table ``kumaari``, model ``kumari``, but
``ਦਾਸ``: table ``das``, model ``daas``), which is why it does not accumulate.
"""

from __future__ import annotations

import gzip
import os
from functools import lru_cache
from pathlib import Path

from .logging import get_logger
from .normalize import NORMALIZER_VERSION, gaz_key, strip_edge_noise
from .resources import local_data_path, try_resolve_data

logger = get_logger()

#: Filename inside each per-language data directory.
LOOKUP_FILE = "lookup.tsv.gz"

#: Bump when the file layout changes incompatibly.
FORMAT_VERSION = 1

#: Directories a table may exist for. Asking for any other language returns
#: ``None`` without touching the network -- otherwise a caller transliterating,
#: say, Tamil would pay a failed Hugging Face round trip to learn nothing.
#:
#: Membership means "look for it", not "it is there". Use :data:`DOWNLOADABLE`
#: for the stronger claim.
SHIPPED = frozenset({"hindi_to_english", "punjabi_to_english"})

#: Directories whose table can be fetched from the model repo.
#:
#: **Empty, and deliberately so.** The tables are derived from
#: ``data/hindi.csv.gz`` (which blends CC-BY-NC IIT Bombay pairs) and
#: ``data/punjabi.csv.gz`` (from the IRB-restricted electoral-roll deposit), so
#: they are not ours to redistribute. Build one locally in a few seconds::
#:
#:     uv run --group train python training/build_lookup.py --lang punjabi
#:
#: Until this is non-empty, a fresh install has no lookup backend, and saying so
#: here is what stops :func:`indicate.supported` advertising one. It also saves
#: every fresh install a guaranteed-404 round trip per language per process.
#: ``tests/e2e/test_hf_contract.py`` asserts that whatever is listed here is
#: actually present in the repo.
DOWNLOADABLE: frozenset[str] = frozenset()

_HEADER_PREFIX = "#"
_CACHE: dict[str, Lookup | None] = {}


# Bounded cache: corpus tokens are Zipf-distributed, so the same few thousand
# words are keyed over and over. Uncached this is the throughput bottleneck
# (412k tok/s vs 2.4M) because str.strip against a 40-character set is not free.
@lru_cache(maxsize=1 << 17)
def lookup_key(word: str) -> str:
    """Return the table key for a surface word.

    Build time and query time must agree exactly or every lookup misses, so both
    go through this one function.

    Args:
        word: A token as it appears in text.

    Returns:
        The normalized key, or ``""`` when nothing lookup-worthy remains.
    """
    return gaz_key(strip_edge_noise(word))


#: How many keys to check before trusting that a whole table is normalized.
_VERIFY_SAMPLE = 1024


def _keys_look_normalized(table: dict[str, str]) -> bool:
    """Report whether a sample of keys is already in canonical form.

    Args:
        table: Freshly parsed key to romanization mapping.

    Returns:
        True if every sampled key is its own :func:`lookup_key`.
    """
    for index, key in enumerate(table):
        if index >= _VERIFY_SAMPLE:
            return True
        if lookup_key(key) != key:
            return False
    return True


class Lookup:
    """An immutable word-level romanization table."""

    def __init__(self, table: dict[str, str], meta: dict[str, str]) -> None:
        """Store an already-parsed table.

        Args:
            table: Normalized key to romanization.
            meta: Header fields from the file.
        """
        self._table = table
        self._meta = meta

    @property
    def convention(self) -> str:
        """Romanization style this table speaks, e.g. ``"roll"``."""
        return self._meta.get("convention", "unknown")

    @property
    def meta(self) -> dict[str, str]:
        """The table's header fields."""
        return dict(self._meta)

    def __len__(self) -> int:
        """Number of keys in the table."""
        return len(self._table)

    def get(self, word: str) -> str | None:
        """Return the romanization for ``word``, or ``None`` on a miss.

        Args:
            word: A token as it appears in text.

        Returns:
            The stored romanization, or ``None``.
        """
        key = lookup_key(word)
        if not key:
            return None
        return self._table.get(key)

    @classmethod
    def from_path(cls, path: Path) -> Lookup | None:
        """Read a table from a gzipped TSV file.

        A table is optional infrastructure, so every failure disables it rather
        than raising: a missing file, a corrupt file, or a table built with a
        different normalizer, whose keys would silently never match.

        Args:
            path: Path to the ``lookup.tsv.gz`` file.

        Returns:
            The table, or ``None`` if it could not be used.
        """
        meta: dict[str, str] = {}
        table: dict[str, str] = {}
        try:
            # Streamed, not read().splitlines(): the latter holds the whole
            # decompressed file *and* a 287k-element list of every line alive at
            # once, which roughly doubles peak RSS for no gain.
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for raw in handle:
                    line = raw.rstrip("\n")
                    if not line:
                        continue
                    if line.startswith(_HEADER_PREFIX):
                        field, _, value = line[1:].partition("\t")
                        meta[field] = value
                        continue
                    key, tab, latin = line.partition("\t")
                    if not tab or not latin or not key:
                        continue  # malformed body line
                    table[key] = latin
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            logger.debug(f"lookup table unavailable at {path}: {exc}")
            return None

        # The builder writes keys through lookup_key, so re-normalizing all of
        # them costs 0.97s of a 1.40s load and changes nothing. Verify a sample
        # instead, and only pay the full pass if the file was not built that way
        # (hand-edited, or produced by something else).
        if not _keys_look_normalized(table):
            table = {k: v for key, v in table.items() if (k := lookup_key(key))}

        stored = meta.get("normalizer")
        if stored != str(NORMALIZER_VERSION):
            logger.warning(
                f"lookup table at {path} was built with normalizer {stored!r}, "
                f"expected {NORMALIZER_VERSION}; disabling it rather than "
                f"serving keys that can never match"
            )
            return None

        logger.debug(f"loaded {len(table):,} lookup keys from {path}")
        return cls(table, meta)

    @classmethod
    def load(cls, subdir: str) -> Lookup | None:
        """Load the table for a language directory, caching the result.

        Args:
            subdir: Per-language directory under ``indicate/data``, e.g.
                ``"punjabi_to_english"``.

        Returns:
            The table, or ``None`` when there is none to be had.
        """
        if subdir in _CACHE:
            return _CACHE[subdir]
        if subdir not in SHIPPED:
            _CACHE[subdir] = None
            return None
        # A local build is the normal case. Only reach for the network when the
        # table is actually published, which today is never -- see DOWNLOADABLE.
        local = local_data_path(subdir, LOOKUP_FILE)
        path = (
            local
            if os.path.exists(local)
            else (
                try_resolve_data(subdir, LOOKUP_FILE)
                if subdir in DOWNLOADABLE
                else None
            )
        )
        table = cls.from_path(Path(path)) if path else None
        _CACHE[subdir] = table
        return table


def clear_cache() -> None:
    """Forget every loaded table. Intended for tests."""
    _CACHE.clear()
